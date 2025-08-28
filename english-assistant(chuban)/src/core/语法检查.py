from language_tool_python import LanguageTool
def analyze_grammar(text):
    """使用LanguageTool进行语法分析"""
    # 使用 LanguageTool 的公共 API
    tool = LanguageTool('en-US', remote_server='https://api.languagetool.org')
    matches = tool.check(text)

    if not matches:
        return {"status": "success", "message": "语法正确！"}

    report = {
        "error_count": len(matches),
        "errors": []
    }

    for match in matches:
        report["errors"].append({
            "rule_id": match.ruleId,
            "message": match.message,
            "context": match.context,
            "replacements": list(match.replacements)
        })

    return report