"""
Test Analysis Script
Analyzes all test files for potential issues without running them
"""
import ast
import sys
from pathlib import Path
from typing import List, Dict, Any
import importlib.util


class TestAnalyzer:
    """Analyze test files for potential issues"""
    
    def __init__(self, tests_dir: Path):
        self.tests_dir = tests_dir
        self.issues: List[Dict[str, Any]] = []
        self.config_files: List[Path] = []
        
    def analyze_all_tests(self) -> Dict[str, Any]:
        """Analyze all test files"""
        test_files = list(self.tests_dir.glob("test_*.py"))
        
        results = {
            "total_tests": len(test_files),
            "files_analyzed": [],
            "issues": [],
            "missing_configs": [],
            "syntax_errors": [],
            "import_errors": []
        }
        
        for test_file in test_files:
            file_result = self.analyze_test_file(test_file)
            results["files_analyzed"].append(file_result)
            
            if file_result.get("syntax_error"):
                results["syntax_errors"].append(file_result)
            if file_result.get("import_errors"):
                results["import_errors"].append(file_result)
            if file_result.get("missing_configs"):
                results["missing_configs"].extend(file_result["missing_configs"])
            if file_result.get("issues"):
                results["issues"].extend(file_result["issues"])
        
        return results
    
    def analyze_test_file(self, test_file: Path) -> Dict[str, Any]:
        """Analyze a single test file"""
        result = {
            "file": str(test_file),
            "syntax_error": None,
            "import_errors": [],
            "missing_configs": [],
            "issues": [],
            "test_count": 0
        }
        
        # Check syntax
        try:
            with open(test_file, 'r', encoding='utf-8') as f:
                source = f.read()
                tree = ast.parse(source, filename=str(test_file))
        except SyntaxError as e:
            result["syntax_error"] = {
                "line": e.lineno,
                "message": str(e),
                "text": e.text
            }
            return result
        
        # Check for test classes and methods
        test_classes = []
        test_methods = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                if node.name.startswith("Test"):
                    test_classes.append(node.name)
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef) and item.name.startswith("test_"):
                            test_methods.append(item.name)
            elif isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                test_methods.append(node.name)
        
        result["test_count"] = len(test_methods)
        result["test_classes"] = test_classes
        
        # Check for config file references
        # Update path: tests/ is now the current directory, so go up one level for demo_data
        demo_data_dir = test_file.parent.parent / "demo_data"
        config_refs = self.find_config_references(source, demo_data_dir)
        result["missing_configs"] = [c for c in config_refs if not c["exists"]]
        
        # Check imports
        imports = self.extract_imports(tree)
        result["imports"] = imports
        
        # Check for common issues
        issues = self.check_common_issues(source, tree)
        result["issues"] = issues
        
        return result
    
    def find_config_references(self, source: str, demo_data_dir: Path) -> List[Dict[str, Any]]:
        """Find references to config files in source code"""
        config_refs = []
        
        # Look for common patterns
        import re
        
        # Pattern 1: Path(...) / 'demo_data' / 'config_*.yaml'
        pattern1 = r"['\"]([^'\"]*config[^'\"]*\.yaml)['\"]"
        matches1 = re.findall(pattern1, source)
        
        # Pattern 2: 'demo_data/config_*.yaml'
        pattern2 = r"demo_data[/\\]([^'\"]*\.yaml)"
        matches2 = re.findall(pattern2, source)
        
        all_configs = set(matches1 + matches2)
        
        for config_name in all_configs:
            # Try different paths
            possible_paths = [
                demo_data_dir / config_name,
                demo_data_dir / config_name.split('/')[-1],
                demo_data_dir / config_name.split('\\')[-1],
            ]
            
            found = False
            for path in possible_paths:
                if path.exists():
                    config_refs.append({
                        "name": config_name,
                        "path": str(path),
                        "exists": True
                    })
                    found = True
                    break
            
            if not found:
                config_refs.append({
                    "name": config_name,
                    "path": str(demo_data_dir / config_name),
                    "exists": False
                })
        
        return config_refs
    
    def extract_imports(self, tree: ast.AST) -> List[str]:
        """Extract import statements"""
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    imports.append(f"{module}.{alias.name}")
        
        return imports
    
    def check_common_issues(self, source: str, tree: ast.AST) -> List[str]:
        """Check for common issues in test code"""
        issues = []
        
        # Check if pytest is imported
        has_pytest = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "pytest":
                        has_pytest = True
            elif isinstance(node, ast.ImportFrom):
                if node.module == "pytest":
                    has_pytest = True
        
        if not has_pytest and "test_" in source:
            issues.append("pytest not imported but test functions found")
        
        # Check for CliRunner usage
        has_cli_runner = "CliRunner" in source
        if has_cli_runner and "from click.testing import CliRunner" not in source:
            issues.append("CliRunner used but may not be imported correctly")
        
        # Check for proper test class structure
        has_test_class = False
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
                has_test_class = True
                break
        
        # Check for if __name__ == '__main__' blocks
        has_main_block = False
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                if isinstance(node.test, ast.Compare):
                    if isinstance(node.test.left, ast.Name) and node.test.left.id == "__name__":
                        has_main_block = True
        
        return issues
    
    def print_report(self, results: Dict[str, Any]):
        """Print analysis report"""
        print("=" * 80)
        print("TEST ANALYSIS REPORT")
        print("=" * 80)
        print(f"\nTotal test files analyzed: {results['total_tests']}")
        print(f"Files with syntax errors: {len(results['syntax_errors'])}")
        print(f"Files with import issues: {len(results['import_errors'])}")
        print(f"Missing config files: {len(results['missing_configs'])}")
        print(f"Total issues found: {len(results['issues'])}")
        
        if results['syntax_errors']:
            print("\n" + "=" * 80)
            print("SYNTAX ERRORS:")
            print("=" * 80)
            for error in results['syntax_errors']:
                print(f"\n  File: {error['file']}")
                print(f"    Line {error['syntax_error']['line']}: {error['syntax_error']['message']}")
        
        if results['missing_configs']:
            print("\n" + "=" * 80)
            print("MISSING CONFIG FILES:")
            print("=" * 80)
            for config in results['missing_configs']:
                print(f"  {config['name']} -> {config['path']}")
        
        if results['import_errors']:
            print("\n" + "=" * 80)
            print("IMPORT ISSUES:")
            print("=" * 80)
            for error in results['import_errors']:
                print(f"  {error['file']}: {error.get('import_errors', [])}")
        
        print("\n" + "=" * 80)
        print("DETAILED FILE ANALYSIS:")
        print("=" * 80)
        for file_result in results['files_analyzed']:
            print(f"\n  {Path(file_result['file']).name}:")
            print(f"    Test classes: {len(file_result.get('test_classes', []))}")
            print(f"    Test methods: {file_result['test_count']}")
            if file_result.get('issues'):
                print(f"    Issues: {', '.join(file_result['issues'])}")
            if file_result.get('missing_configs'):
                print(f"    Missing configs: {len(file_result['missing_configs'])}")


def main():
    """Main entry point"""
    # Update path: tests/ is now the current directory
    tests_dir = Path(__file__).parent
    
    if not tests_dir.exists():
        print(f"Error: Tests directory not found: {tests_dir}")
        sys.exit(1)
    
    analyzer = TestAnalyzer(tests_dir)
    results = analyzer.analyze_all_tests()
    analyzer.print_report(results)
    
    # Return exit code based on issues found
    if results['syntax_errors'] or results['missing_configs']:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
