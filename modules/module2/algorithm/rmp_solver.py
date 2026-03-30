import pulp
import logging
from typing import Dict, List, Tuple
from collections import defaultdict
import numpy as np

logger = logging.getLogger(__name__)

class RMPSolver:
    """
    Restricted Master Problem (RMP) Solver for Branch and Price.
    This formulates the Arc-Flow MILP on a subset of the column pool.
    """
    def __init__(self, c_box: Dict[int, float], c_install: Dict[int, float], 
                 c1: float, c2: float, c3: float, 
                 Q_box_inv: Dict[int, int], N_max: int, 
                 Q_substation: int, substation_coord: Tuple[float, float]):
        self.c_box = c_box
        self.c_install = c_install
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.Q_box_inv = Q_box_inv
        self.N_max = N_max
        self.Q_substation = Q_substation
        self.substation_coord = substation_coord

    def _build_solver(self, is_relaxation: bool):
        """Pick best available MILP solver with graceful fallback."""
        # Prefer commercial solvers when installed.
        # Keep msg=0 to avoid noisy benchmark logs.
        candidates = [
            ("GUROBI", lambda: pulp.GUROBI(msg=0)),
            ("CPLEX_PY", lambda: pulp.CPLEX_PY(msg=0)),
            ("SCIP_PY", lambda: pulp.SCIP_PY(msg=0)),
            ("HiGHS", lambda: pulp.HiGHS(msg=0)),
            ("PULP_CBC_CMD", lambda: pulp.PULP_CBC_CMD(msg=0, timeLimit=120 if not is_relaxation else 60)),
        ]

        for name, factory in candidates:
            try:
                solver = factory()
                if solver is not None and solver.available():
                    logger.info(f"RMP using solver: {name}")
                    return solver
            except Exception:
                continue

        logger.warning("No preferred solver available; fallback to default CBC")
        return pulp.PULP_CBC_CMD(msg=0, timeLimit=120 if not is_relaxation else 60)

    def build_and_solve_rmp(self, inverters: List[Dict], candidate_boxes: List[Tuple[float, float]],
                            column_pool: List[Dict], edges_info: Dict, is_relaxation: bool = True,
                            fixed_to_one: set = None, fixed_to_zero: set = None) -> Dict:
        """
        Builds and solves the Arc-Flow RMP.
        If is_relaxation is True, the problem is solved as an LP (Linear Program) to obtain dual variables.
        If is_relaxation is False, the problem is solved as a MILP to obtain integer solutions.

        Parameters
        ----------
        fixed_to_one : set, optional
            Set of box indices whose y variable is fixed to 1 (B&B branching).
        fixed_to_zero : set, optional
            Set of box indices whose y variable is fixed to 0 (B&B branching).
        """
        if fixed_to_one is None:
            fixed_to_one = set()
        if fixed_to_zero is None:
            fixed_to_zero = set()
        logger.info(f"Building RMP with {len(column_pool)} columns, {len(candidate_boxes)} box combinations, relaxation={is_relaxation}")
        
        prob_type = pulp.LpMinimize
        prob = pulp.LpProblem("RMP", prob_type)
        cat = pulp.LpContinuous if is_relaxation else pulp.LpBinary

        # Variables
        alpha_vars = {path["id"]: pulp.LpVariable(f"alpha_{path['id']}", 0, 1, cat) for path in column_pool}
        beta_vars = {edge: pulp.LpVariable(f"beta_{edge[0]}_{edge[1]}_{edge[2]}_{edge[3]}", 0, 1, cat) for edge in edges_info}
        
        gamma_vars = {}
        for inv in inverters:
            for b_idx, box in enumerate(candidate_boxes):
                gamma_vars[(inv["id"], b_idx)] = pulp.LpVariable(f"gamma_{inv['id']}_{b_idx}", 0, 1, cat)
                
        y_vars = {b_idx: pulp.LpVariable(f"y_{b_idx}", 0, 1, cat) for b_idx in range(len(candidate_boxes))}
        z_vars = {b_idx: pulp.LpVariable(f"z_{b_idx}", 0, 1, cat) for b_idx in range(len(candidate_boxes))} # 0: small, 1: large

        # Apply B&B branching constraints
        for b in fixed_to_one:
            if b in y_vars:
                y_vars[b].lowBound = 1
        for b in fixed_to_zero:
            if b in y_vars:
                y_vars[b].upBound = 0
        
        # Objective Function
        cost_box = pulp.lpSum([
            y_vars[b] * (self.c_box[0] + self.c_install[0]) + 
            z_vars[b] * ((self.c_box[1] + self.c_install[1]) - (self.c_box[0] + self.c_install[0]))
            for b in range(len(candidate_boxes))
        ])
        
        cost_cable_c2 = pulp.lpSum([
            alpha_vars[p["id"]] * p["length"] * self.c2 for p in column_pool if p["type"] == "box_to_sub"
        ])
        
        cost_trench = pulp.lpSum([
            beta_vars[edge] * info["length"] * self.c3 for edge, info in edges_info.items()
        ])
        
        # NOTE: c1 (DC cable) is constant and added in the final aggregator usually, 
        # but path costs for 'dc' can be explicitly modeled if variable
        cost_cable_c1 = pulp.lpSum([
            alpha_vars[p["id"]] * p["length"] * self.c1 for p in column_pool if p["type"] == "inv_to_box"
        ])
        
        prob += cost_box + cost_cable_c2 + cost_cable_c1 + cost_trench
        
        # Constraint 1: Unique assignment
        for inv_idx, inv in enumerate(inverters):
            prob += pulp.lpSum(gamma_vars[(inv["id"], b)] for b in range(len(candidate_boxes))) == 1, f"Unique_Assign_{inv['id']}"
            
        # Constraint 2: Path-assignment consistency
        for inv_idx, inv in enumerate(inverters):
            for b_idx in range(len(candidate_boxes)):
                # Note: path_dict has inv_idx as list index, we use it directly
                relevant_paths = [p["id"] for p in column_pool if p.get("inv_idx") == inv_idx and p.get("box_idx") == b_idx and p["type"] == "inv_to_box"]
                prob += pulp.lpSum(alpha_vars[p_id] for p_id in relevant_paths) == gamma_vars[(inv["id"], b_idx)], f"Consistency_{inv['id']}_{b_idx}"
                
        # Constraint 3: Substation connectivity
        for b_idx in range(len(candidate_boxes)):
            relevant_paths = [p["id"] for p in column_pool if p.get("box_idx") == b_idx and p["type"] == "box_to_sub"]
            prob += pulp.lpSum(alpha_vars[p_id] for p_id in relevant_paths) == y_vars[b_idx], f"Substation_Routing_{b_idx}"
            
        # Constraint 4: Trench co-routing (alpha <= beta) & capacity (sum(alpha) <= N_max * beta)
        edge_to_paths = defaultdict(list)
        for p in column_pool:
            for edge in p["edges"]:
                edge_to_paths[edge].append(p["id"])
                
        for edge in edges_info:
            prob += pulp.lpSum(alpha_vars[p_id] for p_id in edge_to_paths[edge]) <= self.N_max * beta_vars[edge], f"Trench_Cap_{edge}"
            for p_id in edge_to_paths[edge]:
                prob += alpha_vars[p_id] <= beta_vars[edge], f"Trench_Ex_{p_id}_{edge}"
                
        # Constraint 5: Box capacity
        cap_small = list(self.Q_box_inv.values())[0]
        cap_large = list(self.Q_box_inv.values())[1]
        for b_idx in range(len(candidate_boxes)):
            # Active indicator bounding
            prob += z_vars[b_idx] <= y_vars[b_idx], f"Box_Size_Cap_{b_idx}"
            prob += pulp.lpSum(gamma_vars[(inv["id"], b_idx)] for inv in inverters) <= cap_small * y_vars[b_idx] + (cap_large - cap_small) * z_vars[b_idx], f"Box_Cap_{b_idx}"

        # Constraint 6: Substation capacity
        prob += pulp.lpSum(gamma_vars[(inv["id"], b_idx)] for inv in inverters for b_idx in range(len(candidate_boxes))) <= self.Q_substation, "Substation_Cap"

        # ================================================================
        # Enhanced Valid Inequalities (Cutting Planes)
        # ================================================================
        n_boxes = len(candidate_boxes)
        n_inv = len(inverters)
        cap_small = list(self.Q_box_inv.values())[0]

        # (a) Symmetry breaking: order active boxes by index
        for b in range(n_boxes - 1):
            prob += y_vars[b] >= y_vars[b + 1], f"Symmetry_{b}"

        # (b) Minimum box count: ceil(n_inv / max_capacity) boxes required
        min_boxes = int(np.ceil(n_inv / max(self.Q_box_inv.values())))
        prob += pulp.lpSum(y_vars[b] for b in range(n_boxes)) >= min_boxes, "Min_Boxes"

        # (c) Capacity cuts: for any subset of inverters needing more than one box
        # Use simple pairwise capacity cuts for groups of cap+1 inverters
        if n_inv > cap_small and n_boxes > 1:
            for b in range(n_boxes):
                prob += pulp.lpSum(gamma_vars[(inv["id"], b)] for inv in inverters) <= \
                        cap_small * (1 - z_vars[b]) + list(self.Q_box_inv.values())[1] * z_vars[b], \
                        f"Cap_Cut_{b}"

        # Solve it
        prob.solve(self._build_solver(is_relaxation=is_relaxation))
        
        status = prob.status
        if status != pulp.LpStatusOptimal:
            return {"status": "infeasible", "objective": float('inf'), "prob": prob}
            
        # Get duals if relaxation
        duals = {}
        dual_assignment = {}
        dual_trench = {}
        if is_relaxation:
            for name, c in prob.constraints.items():
                duals[name] = c.pi
            # Build structured duals for pricing subproblem
            for inv in inverters:
                c_name = f"Unique_Assign_{inv['id']}"
                dual_assignment[inv["id"]] = duals.get(c_name, 0.0)
            for edge in edges_info:
                c_name = f"Trench_Cap_{edge}"
                dual_trench[edge] = duals.get(c_name, 0.0)

        y_values = {b: y_vars[b].varValue or 0.0 for b in y_vars}

        return {
            "status": "optimal",
            "objective": pulp.value(prob.objective),
            "duals": duals,
            "dual_assignment": dual_assignment,
            "dual_trench": dual_trench,
            "alpha": {p_id: alpha_vars[p_id].varValue for p_id in alpha_vars},
            "beta": {e: beta_vars[e].varValue for e in beta_vars},
            "gamma": {k: gamma_vars[k].varValue for k in gamma_vars},
            "y": y_values,
            "y_values": y_values,
            "z": {b: z_vars[b].varValue for b in z_vars}
        }
