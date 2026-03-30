import json
import os
import logging
from typing import Dict, Optional

class ConfigManager:
    """
    统一配置管理系统
    支持参数的集中管理和动态调整
    """
    def __init__(self, config_path: str = None):
        """
        初始化配置管理器
        Args:
            config_path: 配置文件路径
        """
        # 默认配置
        self.default_config = {
            "rl": {
                "state_dim": 18,
                "action_dim": 7,
                "num_objectives": 3,
                "gamma": 0.98,
                "epsilon": 0.1,
                "epsilon_decay": 0.995,
                "epsilon_min": 0.01,
                "temperature": 1.0,
                "temperature_decay": 0.99,
                "temperature_min": 0.1,
                "batch_size": 32,
                "memory_capacity": 10000,
                "use_dueling": True,
                "exploration_strategy": "boltzmann",
                "use_adaptive_weights": True
            },
            "cable": {
                "r_c_min": 0.012,
                "r_c_max": 0.04,
                "r_c_step": 0.0005
            },
            "optimization": {
                "max_iter_small": 100,
                "max_iter_medium": 150,
                "max_iter_large": 200,
                "heuristic_search_frequency": 50
            },
            "logging": {
                "level": "INFO",
                "log_file": None
            },
            "visualization": {
                "enable": True,
                "save_dir": "visualizations"
            }
        }
        
        # 加载配置文件
        self.config = self.default_config.copy()
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
        
    def load_config(self, config_path: str):
        """
        加载配置文件
        Args:
            config_path: 配置文件路径
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                loaded_config = json.load(f)
                # 递归更新配置
                self._update_config(self.config, loaded_config)
            logging.info(f"成功加载配置文件: {config_path}")
        except Exception as e:
            logging.error(f"加载配置文件失败: {e}")
    
    def save_config(self, config_path: str):
        """
        保存配置文件
        Args:
            config_path: 配置文件路径
        """
        try:
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            logging.info(f"成功保存配置文件: {config_path}")
        except Exception as e:
            logging.error(f"保存配置文件失败: {e}")
    
    def _update_config(self, target: Dict, source: Dict):
        """
        递归更新配置
        Args:
            target: 目标配置
            source: 源配置
        """
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._update_config(target[key], value)
            else:
                target[key] = value
    
    def get(self, key: str, default=None):
        """
        获取配置值
        Args:
            key: 配置键，支持点号分隔的路径
            default: 默认值
        Returns:
            配置值
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value):
        """
        设置配置值
        Args:
            key: 配置键，支持点号分隔的路径
            value: 配置值
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def get_rl_config(self) -> Dict:
        """
        获取RL相关配置
        Returns:
            RL配置字典
        """
        return self.config.get("rl", {})
    
    def get_cable_config(self) -> Dict:
        """
        获取电缆相关配置
        Returns:
            电缆配置字典
        """
        return self.config.get("cable", {})
    
    def get_optimization_config(self) -> Dict:
        """
        获取优化相关配置
        Returns:
            优化配置字典
        """
        return self.config.get("optimization", {})

# 创建全局配置管理器实例
config_manager = ConfigManager()