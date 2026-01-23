#!/usr/bin/env python3
"""
Validation script for habit package refactoring.

Checks if commands follow the standard patterns:
- Uses ConfigClass.from_file() instead of yaml.safe_load()
- Uses ServiceConfigurator.create_*() instead of direct instantiation
- Proper error handling and logging
"""

import re
import sys
from pathlib import Path
from typing import List, Tuple


class RefactoringValidator:
    """Validates refactoring compliance for habit commands."""

    def __init__(self, habit_root: Path):
        self.habit_root = habit_root
        self.commands_dir = habit_root / "cli_commands" / "commands"

    def find_command_files(self) -> List[Path]:
        """Find all command files."""
        return list(self.commands_dir.glob("cmd_*.py"))

    def check_config_loading(self, content: str) -> Tuple[bool, str]:
        """Check if file uses proper config loading pattern."""
        # ✅ Good patterns
        if re.search(r'ConfigClass\.from_file\(', content):
            return True, "Uses ConfigClass.from_file()"

        # ❌ Bad patterns
        if re.search(r'yaml\.safe_load\(', content):
            return False, "Still uses yaml.safe_load() - should use ConfigClass.from_file()"

        return False, "No config loading detected"

    def check_service_creation(self, content: str) -> Tuple[bool, str]:
        """Check if file uses proper service creation pattern."""
        # ✅ Good patterns
        if re.search(r'ServiceConfigurator\.create_\w+\(', content):
            return True, "Uses ServiceConfigurator factory methods"

        # ❌ Bad patterns - direct instantiation
        direct_patterns = [
            r'\w+Workflow\(.*config.*\)',
            r'\w+Service\(.*config.*\)',
            r'\w+Processor\(.*config.*\)',
        ]

        for pattern in direct_patterns:
            if re.search(pattern, content):
                return False, f"Direct instantiation detected: {pattern}"

        return False, "No service creation detected"

    def check_imports(self, content: str) -> Tuple[bool, str]:
        """Check if required imports are present."""
        required_imports = [
            "from habit.core.common.service_configurator import ServiceConfigurator",
        ]

        missing = []
        for imp in required_imports:
            if imp not in content:
                missing.append(imp)

        if not missing:
            return True, "All required imports present"

        return False, f"Missing imports: {', '.join(missing)}"

    def validate_file(self, file_path: Path) -> dict:
        """Validate a single command file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        results = {
            'file': file_path.name,
            'config_loading': self.check_config_loading(content),
            'service_creation': self.check_service_creation(content),
            'imports': self.check_imports(content),
        }

        # Overall compliance
        checks = [results[k][0] for k in ['config_loading', 'service_creation', 'imports']]
        results['compliant'] = all(checks)

        return results

    def validate_all(self) -> List[dict]:
        """Validate all command files."""
        results = []
        for cmd_file in self.find_command_files():
            try:
                result = self.validate_file(cmd_file)
                results.append(result)
            except Exception as e:
                results.append({
                    'file': cmd_file.name,
                    'error': str(e),
                    'compliant': False
                })

        return results

    def print_report(self, results: List[dict]):
        """Print validation report."""
        print("🔍 Habit Package Refactoring Validation Report")
        print("=" * 60)

        compliant = [r for r in results if r.get('compliant', False)]
        non_compliant = [r for r in results if not r.get('compliant', False)]

        print(f"📊 Summary: {len(compliant)}/{len(results)} commands compliant")
        print()

        if non_compliant:
            print("❌ Non-compliant commands:")
            for result in non_compliant:
                print(f"  • {result['file']}")
                if 'error' in result:
                    print(f"    Error: {result['error']}")
                else:
                    for check_name, (passed, message) in result.items():
                        if check_name in ['file', 'compliant']:
                            continue
                        status = "✅" if passed else "❌"
                        print(f"    {status} {check_name}: {message}")
                print()
        else:
            print("🎉 All commands are compliant!")

        if compliant:
            print("✅ Compliant commands:")
            for result in compliant:
                print(f"  • {result['file']}")

    def generate_migration_status(self, results: List[dict]) -> str:
        """Generate updated migration status markdown."""
        template = """# Commands 模块迁移状态检查报告

## 📋 标准模式（参考 cmd_habitat.py）

标准模式应包含：
1. ✅ 使用 `ConfigClass.from_file(config_file)` 加载配置（类型安全，路径自动解析）
2. ✅ 使用 `ServiceConfigurator` 创建服务（依赖注入）
3. ✅ 统一的错误处理和日志记录

## ✅ 已符合标准模式的命令

"""

        compliant = [r for r in results if r.get('compliant', False)]
        non_compliant = [r for r in results if not r.get('compliant', False)]

        for result in compliant:
            template += f"### {result['file']} ✅\n"
            template += "- ✅ 使用标准配置加载模式\n"
            template += "- ✅ 使用 ServiceConfigurator\n\n"

        if non_compliant:
            template += "## ❌ 仍需要更新的命令\n\n"
            for result in non_compliant:
                template += f"### {result['file']} ❌\n"
                if 'error' in result:
                    template += f"**错误**: {result['error']}\n\n"
                else:
                    issues = []
                    for check_name, (passed, message) in result.items():
                        if check_name in ['file', 'compliant'] or passed:
                            continue
                        issues.append(message)
                    if issues:
                        template += "**需要修改**:\n"
                        for issue in issues:
                            template += f"- {issue}\n"
                        template += "\n"

        return template


def main():
    """Main entry point."""
    habit_root = Path(__file__).parent.parent.parent / "habit"

    if not habit_root.exists():
        print(f"Error: Habit root not found at {habit_root}")
        sys.exit(1)

    validator = RefactoringValidator(habit_root)
    results = validator.validate_all()
    validator.print_report(results)

    # Optionally update migration status
    if len(sys.argv) > 1 and sys.argv[1] == "--update-status":
        status_content = validator.generate_migration_status(results)
        status_file = habit_root / "cli_commands" / "COMMANDS_MIGRATION_STATUS.md"
        with open(status_file, 'w', encoding='utf-8') as f:
            f.write(status_content)
        print(f"\n📝 Updated migration status: {status_file}")


if __name__ == "__main__":
    main()