"""
真正的分支定价算法主控文件（重写版）
整合 RMP求解器, Pricing子问题, ColumnManager, 树搜索(Branch and Bound), 和 Lagrangian 松弛。
"""

import logging
import time
import numpy as np
from typing import Dict, List, Tuple
from .base_components import CoordinateHelper, CandidateSiteGenerator, PathFactory
from .pricing_subproblem import PricingSubproblem
from .column_manager import ColumnManager
from .rmp_solver import RMPSolver
from .result_formatter import ResultFormatter
from .lagrangian import LagrangianRelaxation
from .bb_tree import BranchAndBoundTree, BBNode
from .heuristics import MatheuristicFallback

logger = logging.getLogger(__name__)

class BranchAndPrice:
    """
    Branch and Price Master Orchestrator.
    Manages the overall Column Generation loop and Branch & Bound tree traversal.
    """
    def __init__(self, instance_data: Dict, module1_output: Dict):
        self.instance_id = instance_data["instance_info"]["instance_id"]
        self.grid_size = instance_data["terrain_data"]["grid_size"]
        
        # Base Helpers
        self.coord_helper = CoordinateHelper(
            self.grid_size, 
            np.array(instance_data["terrain_data"]["slope_matrix"]),
            np.array(instance_data["terrain_data"]["buildable_matrix"])
        )
        self.path_factory = PathFactory(self.coord_helper)
        
        # Modules specific parameters
        equip = instance_data["equipment_params"]
        transformer = equip["transformer"]
        self.c_box = {0: transformer["c_box"]["1600"], 1: transformer["c_box"]["3200"]}
        self.c_install = {0: transformer["c_install_box"]["1600"], 1: transformer["c_install_box"]["3200"]}
        self.c1 = equip["cable"]["c1"]
        self.c2 = equip["cable"]["c2"]
        self.c3 = equip["cable"]["c3"]
        self.Q_box_inv = {0: 5, 1: 10} # Capacity in inverters
        
        self.Q_substation = equip["substation"]["Q_substation"]
        self.substation_coord = tuple(equip["substation"]["coord"])
        
        # Extraction logic for trench limit
        self.N_max = 4
        for c in instance_data.get("constraint_info", []):
            if c["type"] == "trench_max_cables":
                self.N_max = int(c["value"])
            elif c["type"] == "substation_capacity":
                self.Q_substation = int(c["value"])
        
        # Module 1 Outputs
        self.module1_output = module1_output
        self.inverters = module1_output["zone_summary"]
        self.partition_result = module1_output.get("partition_result", [])
        self.inverter_coords = self._compute_inverter_coords()
        
        # Inject centroids into inverters for heuristic and pricing logic
        for inv, coord in zip(self.inverters, self.inverter_coords):
            inv["centroid"] = coord
            # Handle key mappings
            if "id" not in inv and "inverter_id" in inv:
                inv["id"] = inv["inverter_id"]
        
        # Calculate fixed DC string cable cost from Panel to Inverter (Module 1's domain)
        self.dc_cable_cost = self._compute_dc_cable_cost()
        
        # Generate Candidates
        self.site_generator = CandidateSiteGenerator(
            self.coord_helper, 
            self.inverter_coords, 
            self.substation_coord
        )
        # Note: the old code used site_generator.generate_candidates(), 
        # base_components says generate() perhaps? Let's fix that if needed.
        self.candidate_boxes = self.site_generator.generate()

        
        # Components
        self.column_manager = ColumnManager(
            n_inverters=len(self.inverters), 
            n_sites=len(self.candidate_boxes), 
            max_active_paths=1000
        )
        self.rmp_solver = RMPSolver(
            c_box=self.c_box, c_install=self.c_install, c1=self.c1, c2=self.c2, c3=self.c3,
            Q_box_inv=self.Q_box_inv, N_max=self.N_max, Q_substation=self.Q_substation, substation_coord=self.substation_coord
        )
        self.pricing_solver = PricingSubproblem(
            coord_helper=self.coord_helper, 
            path_factory=self.path_factory,
            inverter_coords=self.inverter_coords, 
            candidate_sites=self.candidate_boxes, 
            substation_coord=self.substation_coord, 
            c2=self.c2, 
            c3=self.c3
        )
        self.heuristic = MatheuristicFallback(
            self.inverters,
            self.candidate_boxes,
            self.Q_box_inv,
            self.path_factory,
            self.substation_coord,
        )

        # Performance monitoring
        self.perf_stats = {
            "cg_iterations": 0,
            "cg_time": 0.0,
            "pricing_time": 0.0,
            "rmp_time": 0.0,
            "bb_nodes": 0,
            "bb_time": 0.0,
            "lagrangian_time": 0.0,
            "total_time": 0.0,
        }
        # Convergence history for visualization
        self.convergence_history: List[Dict] = []

        self.formatter = ResultFormatter(
            inverters=self.inverters, 
            candidate_boxes=self.candidate_boxes,
            equipment_params=equip,
            N_max=self.N_max, 
            grid_size=self.grid_size,
            dc_cable_cost=self.dc_cable_cost
        )
        
        # Data caches
        self.edges_info = {}

    def _compute_inverter_coords(self) -> List[Tuple[float, float]]:
        """计算每个逆变器的坐标（分区面板重心）"""
        coords = []
        for zone in self.inverters:
            zone_id = zone.get("zone_id", zone.get("id"))
            panels = [p for p in self.partition_result if p.get("zone_id") == zone_id]
            if panels:
                avg_row = sum(p["grid_coord"][0] for p in panels) / len(panels)
                avg_col = sum(p["grid_coord"][1] for p in panels) / len(panels)
                x, y = self.coord_helper.grid_to_xy((avg_row, avg_col))
                coords.append(self.coord_helper.align_to_grid((x, y)))
            else:
                coords.append(self.coord_helper.align_to_grid(self.substation_coord))
        return coords

    def _compute_dc_cable_cost(self):
        c1_wan = self.c1 / 10000.0
        total_cost = 0.0
        for zone in self.inverters:
            if "cable_routes" in zone:
                total_len = sum(edge.get("length", 0) for edge in zone["cable_routes"])
                total_cost += total_len * c1_wan
        return total_cost
        
    def _initialize_columns(self, heur_sol: Dict = None):
        """Phase 1: Generate initial set of feasible columns using K-Means heuristic and KNN paths."""
        logger.info("Initializing baseline columns (Warm Start)...")

        # --- K-Means heuristic columns (warm start) ---
        if heur_sol and heur_sol.get("columns"):
            self.column_manager.add_paths(heur_sol["columns"])
            for col in heur_sol["columns"]:
                self._update_edges(col["edges"], col)
            logger.info(f"Added {len(heur_sol['columns'])} K-Means heuristic columns")

        # --- KNN baseline: each inverter → nearest 5 boxes ---
        for inv_idx, inv in enumerate(self.inverters):
            dists = [(b_idx, np.linalg.norm(np.array(inv["centroid"]) - np.array(bc)))
                     for b_idx, bc in enumerate(self.candidate_boxes)]
            dists.sort(key=lambda x: x[1])
            for b_idx, _ in dists[:5]:
                box_coord = self.candidate_boxes[b_idx]
                path_dict = self.path_factory.generate_path(
                    inv["centroid"], box_coord, path_type="inv_to_box", inv_idx=inv_idx, box_idx=b_idx
                )
                p_id = f"init_{inv_idx}_{b_idx}"
                path_dict["id"] = p_id
                path_dict["inv_id"] = inv["id"]
                self.column_manager.add_paths([path_dict])
                self._update_edges(path_dict["edges"], path_dict)

        # --- Substation paths: each box → substation ---
        for b_idx, box_coord in enumerate(self.candidate_boxes):
            path_dict = self.path_factory.generate_path(
                box_coord, self.substation_coord, path_type="box_to_sub", box_idx=b_idx, inv_idx=None
            )
            p_id = f"init_sub_{b_idx}"
            path_dict["id"] = p_id
            self.column_manager.add_paths([path_dict])
            self._update_edges(path_dict["edges"], path_dict)
            
    def _extract_edges(self, path: List[Tuple[float, float]]) -> List[Tuple]:
        return [(path[i][0], path[i][1], path[i+1][0], path[i+1][1]) for i in range(len(path)-1)]
        
    def _update_edges(self, edges: List[Tuple], path_dict: Dict):
        for e in edges:
            if e not in self.edges_info:
                # Fallback to general factory dict if info is strictly in factory.
                self.edges_info[e] = self.path_factory.edges[e]

    def _run_column_generation(self, fixed_to_one: set = None, fixed_to_zero: set = None,
                                max_cg_iter: int = 20) -> Dict:
        """
        Column Generation loop at a single B&B node.
        Includes column clustering/pruning for scalability.
        Returns the final LP relaxation result.
        """
        t_cg = time.time()
        rmp_sol = {"status": "infeasible"}
        for iteration in range(max_cg_iter):
            # --- RMP solve ---
            t_rmp = time.time()
            active_cols = self.column_manager.get_active_paths()
            rmp_sol = self.rmp_solver.build_and_solve_rmp(
                self.inverters, self.candidate_boxes, active_cols, self.edges_info,
                is_relaxation=True, fixed_to_one=fixed_to_one, fixed_to_zero=fixed_to_zero
            )
            self.perf_stats["rmp_time"] += time.time() - t_rmp

            if rmp_sol["status"] != "optimal":
                logger.warning(f"RMP Relaxation Infeasible at CG iter {iteration}")
                self.perf_stats["cg_time"] += time.time() - t_cg
                return rmp_sol

            # Record convergence
            self.convergence_history.append({
                "iteration": self.perf_stats["cg_iterations"] + iteration + 1,
                "objective": rmp_sol["objective"],
                "n_active_paths": len(active_cols),
                "n_total_paths": len(self.column_manager.all_paths),
            })

            # --- Pricing subproblem ---
            t_price = time.time()
            dual_assignment = rmp_sol.get("dual_assignment", {})
            dual_trench = rmp_sol.get("dual_trench", {})

            new_cols = self.pricing_solver.solve(
                dual_assignment=dual_assignment,
                dual_trench=dual_trench,
                active_path_ids=set(p["id"] for p in active_cols),
                all_paths=self.column_manager.all_paths,
                max_new_paths=20
            )
            self.perf_stats["pricing_time"] += time.time() - t_price

            if not new_cols:
                logger.info(f"CG converged after {iteration + 1} iterations.")
                break

            self.column_manager.add_paths(new_cols)
            for col in new_cols:
                self._update_edges(col["edges"], col)

            # --- Periodic column clustering & pruning (every 5 iterations) ---
            if (iteration + 1) % 5 == 0 and len(self.column_manager.all_paths) > 50:
                self.column_manager.cluster_paths()
                # Compute reduced costs for pruning
                rc_map = {}
                for p in self.column_manager.all_paths:
                    if p["type"] == "inv_to_box":
                        rc_map[p["id"]] = self.pricing_solver._compute_reduced_cost(
                            p, dual_assignment, dual_trench)
                self.column_manager.prune_low_potential(rc_map, threshold=0.1)
                self.column_manager.ensure_feasibility()
                logger.info(f"Post-prune: {len(self.column_manager.get_active_paths())} active paths")

        self.perf_stats["cg_iterations"] += iteration + 1
        self.perf_stats["cg_time"] += time.time() - t_cg
        return rmp_sol

    def _compute_lagrangian_bound(self) -> float:
        """Compute Lagrangian lower bound using current column pool."""
        t0 = time.time()
        try:
            # LagrangianRelaxation expects string-keyed ("1600"/"3200") dicts
            # while BranchAndPrice stores integer-keyed (0/1) dicts.
            c_box_lr = {"1600": self.c_box[0], "3200": self.c_box[1]}
            c_install_lr = {"1600": self.c_install[0], "3200": self.c_install[1]}
            Q_box_inv_lr = {1600: self.Q_box_inv[0], 3200: self.Q_box_inv[1]}
            lr = LagrangianRelaxation(
                n_inverters=len(self.inverters),
                n_sites=len(self.candidate_boxes),
                paths=self.column_manager.all_paths,
                edges=self.edges_info,
                Q_box_inv=Q_box_inv_lr,
                N_max=self.N_max,
                c_box=c_box_lr,
                c_install=c_install_lr,
                c2=self.c2,
                c3=self.c3,
            )
            lr_result = lr.solve(max_iterations=20, step_size=0.5)
            lb = lr_result["lower_bound"]
            logger.info(f"Lagrangian LB = {lb:.2f}")
        except Exception as e:
            logger.warning(f"Lagrangian relaxation failed: {e}")
            lb = -float('inf')
        self.perf_stats["lagrangian_time"] += time.time() - t0
        return lb

    def optimize(self, strategy: str = "branch_and_price", time_limit: int = 300) -> Dict:
        """Main Branch and Price Execution Loop."""
        t_total = time.time()
        logger.info(f"Starting Branch and Price for {self.instance_id} with strategy={strategy}")

        # Reset per-run performance counters for reproducible benchmark rows.
        for key in self.perf_stats:
            self.perf_stats[key] = 0 if key.endswith("iterations") or key.endswith("nodes") else 0.0

        # Initialize column pool
        self.column_manager.initialize([])
        self.convergence_history.clear()

        # Phase 0: Heuristic Upper Bound
        heur_sol = self.heuristic.solve_kmeans_heuristic()

        if strategy == "matheuristic":
            self.perf_stats["total_time"] = time.time() - t_total
            result = self.formatter.format_results(heur_sol, self.column_manager.get_active_paths(), self.edges_info)
            result["convergence_history"] = self.convergence_history
            result["perf_stats"] = self.perf_stats
            result["used_fallback"] = False
            result["solve_status"] = heur_sol.get("status", "heuristic_success")
            return result

        # Phase 1: Initialize columns with heuristic warm start
        self._initialize_columns(heur_sol)

        if strategy == "milp":
            self._run_column_generation()
            active_cols = self.column_manager.get_active_paths()
            final_sol = self.rmp_solver.build_and_solve_rmp(
                self.inverters, self.candidate_boxes, active_cols, self.edges_info, is_relaxation=False
            )
            used_fallback = False
            solve_status = final_sol.get("status", "unknown")
            if final_sol["status"] == "infeasible":
                logger.warning("MILP Infeasible, falling back to heuristic.")
                final_sol = heur_sol
                used_fallback = True
                solve_status = "milp_infeasible_fallback_heuristic"
            self.perf_stats["total_time"] = time.time() - t_total
            result = self.formatter.format_results(final_sol, active_cols, self.edges_info)
            result["convergence_history"] = self.convergence_history
            result["perf_stats"] = self.perf_stats
            result["used_fallback"] = used_fallback
            result["solve_status"] = solve_status
            return result

        # ================================================================
        # Phase 2: Branch and Price (CG + B&B + Lagrangian tightening)
        # ================================================================
        t_bb = time.time()
        bb_tree = BranchAndBoundTree(
            n_sites=len(self.candidate_boxes),
            time_limit=time_limit,
            gap_tolerance=0.05,
            max_nodes=200
        )

        # Set heuristic objective as initial upper bound
        heur_obj = heur_sol.get("objective", float('inf'))
        if heur_obj < float('inf'):
            bb_tree.global_ub = heur_obj
            bb_tree.best_integer_solution = heur_sol

        root = bb_tree.create_root()

        # Compute Lagrangian bound at root for tighter initial LB
        lr_lb = self._compute_lagrangian_bound()
        if lr_lb > bb_tree.global_lb:
            bb_tree.global_lb = lr_lb

        while not bb_tree.should_terminate():
            node = bb_tree.get_next_node()
            if node is None:
                break

            logger.info(f"【B&B】Exploring node {node.node_id} (depth={node.depth}, "
                        f"LB={node.lower_bound:.2f}, UB={bb_tree.global_ub:.2f})")

            # Column generation at this node
            lp_result = self._run_column_generation(
                fixed_to_one=node.fixed_to_one,
                fixed_to_zero=node.fixed_to_zero,
                max_cg_iter=20
            )

            if lp_result["status"] != "optimal":
                bb_tree.process_node_result(node, None)
                continue

            # Map RMP result to B&B expected format
            bb_result = {
                "status": "Optimal",
                "objective": lp_result["objective"],
                "y_values": lp_result.get("y_values", lp_result.get("y", {})),
            }

            action = bb_tree.process_node_result(node, bb_result)

            if action == "integer":
                logger.info(f"【B&B】Integer solution at node {node.node_id}, obj={lp_result['objective']:.2f}")
            elif action == "branch":
                left, right = bb_tree.branch(node)
                if left is None:
                    logger.info(f"【B&B】No branching variable at node {node.node_id}, treating as integer.")

        self.perf_stats["bb_time"] = time.time() - t_bb
        self.perf_stats["bb_nodes"] = bb_tree.stats["nodes_explored"]

        # Phase 3: Extract best solution
        summary = bb_tree.get_summary()
        logger.info(f"【B&B 完成】explored={summary['nodes_explored']}, "
                    f"LB={summary['global_lb']:.2f}, UB={summary['global_ub']:.2f}, "
                    f"gap={summary['gap']}, time={summary['elapsed_time']:.1f}s")

        active_cols = self.column_manager.get_active_paths()
        best_fixed_one = set()
        if bb_tree.incumbent_node_id is not None:
            best_node = bb_tree.all_nodes[bb_tree.incumbent_node_id]
            best_fixed_one = best_node.fixed_to_one

        final_sol = self.rmp_solver.build_and_solve_rmp(
            self.inverters, self.candidate_boxes, active_cols, self.edges_info,
            is_relaxation=False, fixed_to_one=best_fixed_one
        )

        used_fallback = False
        solve_status = final_sol.get("status", "unknown")
        if final_sol["status"] == "infeasible":
            logger.warning("Final MILP infeasible, falling back to heuristic.")
            final_sol = heur_sol
            used_fallback = True
            solve_status = "branch_and_price_infeasible_fallback_heuristic"

        self.perf_stats["total_time"] = time.time() - t_total
        result = self.formatter.format_results(final_sol, active_cols, self.edges_info)
        result["convergence_history"] = self.convergence_history
        result["perf_stats"] = self.perf_stats
        result["bb_summary"] = summary
        result["used_fallback"] = used_fallback
        result["solve_status"] = solve_status
        return result
