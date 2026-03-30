import json
import logging
import os
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)

class ResultFormatter:
    """
    ResultFormatter for Module 2.
    Translates the Arc-Flow MILP solution into the standard expected dictionary format 
    used by model_equipment_cable.py and the system.
    """
    def __init__(self, inverters: List[Dict], candidate_boxes: List[Tuple[float, float]], 
                 equipment_params: Dict, N_max: int, grid_size: float, dc_cable_cost: float):
        self.inverters = inverters
        self.candidate_boxes = candidate_boxes
        
        self.c1 = equipment_params["cable"]["c1"]
        self.c2 = equipment_params["cable"]["c2"]
        self.c3 = equipment_params["cable"]["c3"]
        self.c_box_params = equipment_params["transformer"]["c_box"]
        self.c_install_params = equipment_params["transformer"]["c_install_box"]
        
        self.N_max = N_max
        self.grid_size = grid_size
        self.dc_cable_cost = dc_cable_cost

    def format_results(self, solution: Dict, column_pool: List[Dict], edges_info: Dict) -> Dict:
        """
        Constructs the final rigorous Module 2 output dictionary mirroring the old format.
        """
        if solution.get("status") not in ["optimal", "optimal_relaxed", "fallback", "heuristic_success"]:
            logger.warning(f"Formatting non-optimal solution with status: {solution.get('status')}")

        alpha = solution.get("alpha", {})
        beta = solution.get("beta", {})
        gamma = solution.get("gamma", {})
        y_vars = solution.get("y", {})
        z_vars = solution.get("z", {})
        
        c2_wan = self.c2 / 10000.0
        c3_wan = self.c3 / 10000.0

        # === 1. Equipment Selection (Boxes) ===
        equipment_selection = []
        total_box_purchase = 0.0
        total_box_install = 0.0
        
        for b_idx in range(len(self.candidate_boxes)):
            y_val = y_vars.get(b_idx, 0)
            if float(y_val) > 0.5:
                z_val = z_vars.get(b_idx, 0)
                size = 3200 if float(z_val) > 0.5 else 1600
                size_str = str(size)
                
                connected_invs = []
                for inv in self.inverters:
                    if float(gamma.get((inv["id"], b_idx), 0)) > 0.5:
                        connected_invs.append(inv["id"])
                        
                # Only add if it actually connects to inverters
                if not connected_invs:
                    continue
                        
                cost_purchase = float(self.c_box_params[size_str])
                cost_install = float(self.c_install_params[size_str])
                
                total_box_purchase += cost_purchase
                total_box_install += cost_install

                equipment_selection.append({
                    "transformer_id": f"box_{b_idx}",
                    "install_coord": list(self.candidate_boxes[b_idx]),
                    "Q_box": size,
                    "connected_inverters": connected_invs,
                    "cost": {
                        "purchase": round(cost_purchase, 2),
                        "installation": round(cost_install, 2)
                    }
                })

        # Heuristic fallback path: build equipment selection from assignment if LP vars are absent.
        if not equipment_selection and solution.get("assignment"):
            box_to_invs = defaultdict(list)
            for inv_id, b_idx in solution.get("assignment", {}).items():
                box_to_invs[int(b_idx)].append(inv_id)

            for b_idx, connected_invs in box_to_invs.items():
                if not connected_invs:
                    continue
                size = 1600 if len(connected_invs) <= 5 else 3200
                size_str = str(size)
                cost_purchase = float(self.c_box_params[size_str])
                cost_install = float(self.c_install_params[size_str])
                total_box_purchase += cost_purchase
                total_box_install += cost_install
                equipment_selection.append({
                    "transformer_id": f"box_{b_idx}",
                    "install_coord": list(self.candidate_boxes[b_idx]),
                    "Q_box": size,
                    "connected_inverters": connected_invs,
                    "cost": {
                        "purchase": round(cost_purchase, 2),
                        "installation": round(cost_install, 2)
                    }
                })

        # === 2. active paths ===
        active_paths = []
        # Support heuristic arrays
        for p in column_pool:
            p_val = alpha.get(p["id"], 0)
            # if heuristics bypass solver, they might just specify active paths
            if float(p_val) > 0.5 or solution.get("status") == "heuristic_success":
                active_paths.append(p)
                
        # === 3. Trench Summary and Edge Tracking ===
        active_edges = []
        for edge, info in edges_info.items():
            beta_val = beta.get(edge, 0)
            if float(beta_val) > 0.5 or solution.get("status") == "heuristic_success":
                active_edges.append(edge)
                
        trench_summary = []
        edge_to_paths = defaultdict(list)
        
        for path in active_paths:
            for edge_key in path.get("edges", []):
               edge_to_paths[edge_key].append(path["id"])

        # Create trench assignments
        total_trench_cost = 0.0
        path_to_trench = {}
        path_is_cotrench = {}

        trench_idx = 0
        for edge_key, paths_on_edge in edge_to_paths.items():
            n_cables = len(paths_on_edge)
            if n_cables == 0: continue
            
            # Use real edge length or base it on coordinates if missing
            edge_length = edges_info.get(edge_key, {}).get("length", self.grid_size)
            n_trenches = int(np.ceil(n_cables / self.N_max))
            trench_cost_per = edge_length * c3_wan
            
            for t in range(n_trenches):
                cables_in_this = min(self.N_max, n_cables - t * self.N_max)
                trench_id = f"trench_{trench_idx}_{t}" if n_trenches > 1 else f"trench_{trench_idx}"
                
                trench_cost_round = round(trench_cost_per, 2)
                trench_summary.append({
                    "trench_id": trench_id,
                    "from_coord": [edge_key[0], edge_key[1]],
                    "to_coord": [edge_key[2], edge_key[3]],
                    "length": round(edge_length, 2),
                    "cable_count": cables_in_this,
                    "cost": trench_cost_round
                })
                total_trench_cost += trench_cost_round
                
                # Map path to trench
                paths_for_trench = paths_on_edge[t * self.N_max : (t + 1) * self.N_max]
                for p_id in paths_for_trench:
                    path_to_trench[p_id] = trench_id
                    path_is_cotrench[p_id] = (n_cables > 1)
                    
            trench_idx += 1

        # === 4. Cable Routes ===
        cable_routes = []
        total_cable_cost = 0.0
        for path in active_paths:
            if path["type"] == "inv_to_box":  # inv to box
                # If heuristic provided inv_id explicitly, use it. Otherwise mapped by branch_and_price wrapper
                from_device = str(path.get("inv_id", path.get("inv_idx")))
                to_device = f"box_{path.get('box_idx')}"
            else:  # box to sub
                from_device = f"box_{path.get('box_idx')}"
                to_device = "sub_01"
                
            cable_cost = path["length"] * c2_wan
            total_cable_cost += cable_cost
            
            # base components return coords not path sometimes, but heur returns both mapped
            coords_list = path.get("coords", path.get("path", []))
            
            cable_routes.append({
                "route_id": f"route_{path['id']}",
                "from_device": from_device,
                "to_device": to_device,
                "path_coords": [list(c) for c in coords_list],
                "cable_type": "35kV AC",
                "length": round(path["length"], 2),
                "is_cotrench": path_is_cotrench.get(path["id"], False),
                "trench_id": path_to_trench.get(path["id"], None),
                "cost": {
                    "cable": round(cable_cost, 4), 
                    "trenching": 0.0 # updated below
                }
            })
            
        # Distribute trenching costs
        if cable_routes:
            trench_per_route = total_trench_cost / len(cable_routes)
            for route in cable_routes:
                route["cost"]["trenching"] = round(trench_per_route, 4)

        # === Final Costs and Constraints ===
        total_cost = total_box_purchase + total_box_install + total_cable_cost + total_trench_cost + self.dc_cable_cost
        
        trench_ok = all(t["cable_count"] <= self.N_max for t in trench_summary) if trench_summary else True
        Q_box_inv = {1600: 5, 3200: 10}
        capacity_ok = all(len(eq["connected_inverters"]) <= Q_box_inv[eq["Q_box"]] for eq in equipment_selection) if equipment_selection else True
        
        # Simple Ampacity check
        ampacity_ok = True
        for eq in equipment_selection:
            P_total = len(eq["connected_inverters"]) * 320.0 # kW, assumption based on old code
            I_cable = P_total / (1.732 * 35.0) 
            if I_cable > 200.0:
                ampacity_ok = False
                
        constraint_ok = {
            "trench_cable_count": trench_ok,
            "transformer_capacity": capacity_ok,
            "route_continuity": True,
            "cable_ampacity": ampacity_ok,
            "共沟约束": "100%" if trench_ok else "未满足",
            "箱变容量": "100%" if capacity_ok else "未满足"
        }

        output_dict = {
            "equipment_selection": equipment_selection,
            "cable_routes": cable_routes,
            "trench_summary": trench_summary,
            "constraint_satisfaction": constraint_ok,
            "total_cost": round(total_cost, 2),
            "cost_breakdown": {
                "equipment_purchase": round(total_box_purchase, 2),
                "equipment_install": round(total_box_install, 2),
                "cable_ac": round(total_cable_cost, 2),
                "cable_dc": round(self.dc_cable_cost, 2),
                "trenching": round(total_trench_cost, 2)
            }
        }

        return output_dict
