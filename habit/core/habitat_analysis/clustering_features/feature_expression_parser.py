"""
Feature expression parser for parsing complex feature construction expressions
"""

import re
from typing import Dict, List, Any, Callable, Optional, Tuple, Union

class FeatureExpressionParser:
    """Parser for feature construction expressions"""
    def __init__(self):
        """Initialize the parser"""
        pass

    def parse(self, config: Union[str, Dict[str, Any]], params: Optional[Dict[str, Any]] = None) -> Tuple[str, Dict[str, Any], List[Dict[str, Any]]]:
        """Parse a feature construction expression

        Args:
            config: Feature construction expression string or configuration dictionary
                If dictionary, should have format:
                {
                    "method": "kinetic(raw(pre_contrast, p1), raw(LAP, p1), raw(PVP, p2), raw(delay_3min, p3), timestamps)",
                    "params": {
                        "p1": 1.0,
                        "p2": True,
                        "p3": False,
                        "timestamps": "path/to/timestamps.xlsx"
                    }
                }
                也支持单图像处理方式：
                {
                    "method": "raw(pre_contrast)",
                    "params": {}
                }
                也支持其他处理方式：
                {
                    "method": "supervoxel_radiomics(pre_contrast)",
                    "params": {
                        "params_file": "path/to/radiomics_params.yaml"
                    }
                }
            params: Optional dictionary of parameters for the expression

        Returns:
            Tuple[str, Dict[str, Any], List[Dict[str, Any]]]: Tuple containing:
                - cross_image_method: The main method name
                - cross_image_params: Parameters for the cross-image method
                - processing_steps: List of processing steps

        Raises:
            TypeError: If config is not a string or dictionary
            ValueError: If expression format is invalid or config is missing required fields
        """
        # Extract expression and parameters from config
        if isinstance(config, str):
            expression = config
            params = params or {}
        elif isinstance(config, dict):
            if 'method' not in config:
                raise ValueError("Configuration dictionary must contain 'method' field")
            expression = config['method']
            # Get parameters from config's params field
            config_params = config.get('params', {})
            # Merge with provided params, giving priority to provided params
            if params:
                config_params.update(params)
            params = config_params
        elif hasattr(config, 'model_dump'):  # Pydantic model
            config_dict = config.model_dump()
            if 'method' not in config_dict:
                raise ValueError("Configuration must contain 'method' field")
            expression = config_dict['method']
            config_params = config_dict.get('params', {})
            if params:
                config_params.update(params)
            params = config_params
        else:
            raise TypeError(f"Config must be a string, dictionary, or Pydantic model, got {type(config)}")

        # Initialize return values
        cross_image_method = None
        cross_image_params = {}
        processing_steps = []

        # 提取最外层的方法名
        main_method_match = re.match(r'^(\w+)\s*\(', expression)
        if not main_method_match:
            main_method_match = expression.replace('(', '').replace(')', '')
            cross_image_method = main_method_match
        else:
            method_name = main_method_match.group(1)
            cross_image_method = method_name

        # 提取方法括号内的内容
        try:
            inner_expr = self._extract_inner_expression(expression)
        except Exception as e:
            raise ValueError(f"Failed to extract inner expression: {str(e)}")

        # 处理单方法表达式（如raw(image_name)或supervoxel_radiomics(image_name)）
        if not self._contains_function_call(inner_expr):
            # 简单表达式，例如 raw(pre_contrast) 或 supervoxel_radiomics(pre_contrast)
            image_name = inner_expr.strip()

            # 创建处理步骤
            step = {
                'method': method_name,
                'image': image_name,
                'params': {}
            }

            # 添加方法参数
            for param_name, param_value in params.items():
                if param_name != 'image':  # 避免覆盖image参数
                    step['params'][param_name] = param_value

            processing_steps.append(step)
            return cross_image_method, cross_image_params, processing_steps

        # 处理嵌套方法表达式
        # 拆分内部表达式
        parts = self._split_expression(inner_expr)

        # 处理每个部分
        for part in parts:
            part = part.strip()
            if not part:
                continue

            # 处理函数调用
            if self._is_function_call(part):
                # 解析函数和参数
                func_name, func_args = self._parse_function_call(part)

                # 解析参数列表
                arg_list = self._split_expression(func_args)

                if len(arg_list) == 0:
                    raise ValueError(f"Empty arguments for function: {func_name}")

                # 第一个参数应该是图像名称
                image_name = arg_list[0].strip()

                # 创建处理步骤
                step = {
                    'method': func_name,
                    'image': image_name,
                    'params': {}
                }

                # 处理其他参数
                for i, arg in enumerate(arg_list[1:], 1):
                    arg = arg.strip()
                    # 检查参数名是否存在于params字典中
                    if arg in params:
                        # 使用参数名称作为key
                        step['params'][arg] = params[arg]
                    else:
                        # 如果参数名不在params中，尝试直接使用参数值
                        # 对于supervoxel_radiomics等方法，参数名可能就是参数本身
                        # 例如：supervoxel_radiomics(pre_contrast, parameter)中的parameter
                        # 这里我们使用参数名作为key，而不是param{i}
                        step['params'][arg] = arg

                processing_steps.append(step)
            else:
                # 检查是否是参数名称
                if part in params:
                    # 这是主方法的参数
                    cross_image_params[part] = params[part]
                else:
                    # 不是函数调用也不是已知参数，假设是图像名称
                    step = {
                        'method': 'raw',  # 默认使用raw
                        'image': part,
                        'params': {}
                    }
                    processing_steps.append(step)

        return cross_image_method, cross_image_params, processing_steps

    def _extract_inner_expression(self, expression: str) -> str:
        """提取函数调用的内部表达式

        Args:
            expression: 表达式字符串

        Returns:
            str: 内部表达式
        """
        # 找到第一个左括号
        first_paren = expression.find('(')
        if first_paren == -1:
            raise ValueError(f"No opening parenthesis found in expression: {expression}")

        # 找到匹配的右括号
        paren_count = 1
        for i in range(first_paren + 1, len(expression)):
            if expression[i] == '(':
                paren_count += 1
            elif expression[i] == ')':
                paren_count -= 1
                if paren_count == 0:
                    # 返回内部表达式，不包括最外层的括号
                    return expression[first_paren + 1:i]

        raise ValueError(f"Unmatched opening parenthesis in expression: {expression}")

    def _is_function_call(self, expr: str) -> bool:
        """检查表达式是否是函数调用

        Args:
            expr: 表达式字符串

        Returns:
            bool: 如果表达式是函数调用则为True
        """
        return re.match(r'^\w+\s*\(', expr) is not None

    def _contains_function_call(self, expr: str) -> bool:
        """检查表达式是否包含函数调用

        Args:
            expr: 表达式字符串

        Returns:
            bool: 如果表达式包含函数调用则为True
        """
        return re.search(r'\w+\s*\(', expr) is not None

    def _parse_function_call(self, expr: str) -> Tuple[str, str]:
        """解析函数调用，提取函数名和参数

        Args:
            expr: 函数调用表达式

        Returns:
            Tuple[str, str]: 函数名和参数字符串
        """
        match = re.match(r'^(\w+)\s*\((.*)\)$', expr)
        if not match:
            raise ValueError(f"Invalid function call format: {expr}")

        func_name = match.group(1)
        func_args = match.group(2)

        return func_name, func_args

    def _split_expression(self, expression: str) -> List[str]:
        """Split an expression into parts, handling nested parentheses

        Args:
            expression: Expression string to split

        Returns:
            List[str]: List of expression parts
        """
        if not expression:
            return []

        parts = []
        current = ""
        paren_count = 0

        for char in expression:
            if char == '(':
                paren_count += 1
                current += char
            elif char == ')':
                paren_count -= 1
                current += char
                if paren_count < 0:
                    raise ValueError("Unmatched closing parenthesis")
            elif char == ',' and paren_count == 0:
                parts.append(current.strip())
                current = ""
            else:
                current += char

        if paren_count != 0:
            raise ValueError("Unmatched opening parenthesis")

        if current:
            parts.append(current.strip())

        return [p for p in parts if p]