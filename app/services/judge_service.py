import json
import logging

import google.generativeai as genai

from app.core.config import settings

logger = logging.getLogger(__name__)


def _build_prompt(error_log: str, suggestion: dict) -> str:
    fp = suggestion.get("file_patch", {})
    return f"""당신은 시니어 백엔드 엔지니어입니다. 아래 에러와 수정 제안을 검토해주세요.

## 에러 로그
{error_log[:1500]}

## 분석 결과
- 에러 원인: {suggestion.get("error_cause", "")}
- 병목 지점: {suggestion.get("bottleneck", "")}
- 수정 설명: {suggestion.get("suggested_fix", "")}

## 코드 수정
- 파일: {fp.get("file_path", "")}
- Before:
{fp.get("before", "")}
- After:
{fp.get("after", "")}

## 평가 기준
1. 수정이 에러 원인을 실제로 해결하는가?
2. After 코드가 논리적으로 올바른가?
3. 부작용이나 새로운 버그를 유발할 가능성이 있는가?

JSON만 응답하세요 (마크다운 금지):
{{"score": 1~5, "confidence": "high" | "medium" | "low", "reason": "한 문장 평가"}}"""


async def judge_fix(error_log: str, llm_suggestion: str) -> dict | None:
    """Gemini로 LLM 분석 결과의 품질을 평가한다. 실패 시 None 반환."""
    if not settings.gemini_api_key:
        logger.info("[judge] gemini_api_key not set, skipping")
        return None

    try:
        suggestion = json.loads(llm_suggestion)
    except (json.JSONDecodeError, TypeError):
        # JSON 파싱 실패 시 텍스트에서 JSON 블록 추출 시도
        import re
        m = re.search(r"\{.*\}", llm_suggestion, re.DOTALL)
        if not m:
            logger.warning("[judge] suggestion is not valid JSON, skipping")
            return None
        try:
            suggestion = json.loads(m.group())
        except (json.JSONDecodeError, TypeError):
            logger.warning("[judge] suggestion JSON extraction failed, skipping")
            return None

    if not suggestion.get("file_patch", {}).get("file_path"):
        return None

    try:
        genai.configure(api_key=settings.gemini_api_key)
        model = genai.GenerativeModel(settings.gemini_model)
        prompt = _build_prompt(error_log, suggestion)

        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.1,
                max_output_tokens=256,
            ),
        )
        raw = response.text.strip()

        # 마크다운 코드블록 제거
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        result = json.loads(raw.strip())
        score = int(result.get("score", 0))
        confidence = result.get("confidence", "low")
        reason = result.get("reason", "")

        logger.info("[judge] score=%d confidence=%s reason=%s", score, confidence, reason)
        return {"score": score, "confidence": confidence, "reason": reason}

    except Exception as e:
        logger.warning("[judge] failed: %s", e)
        return None
