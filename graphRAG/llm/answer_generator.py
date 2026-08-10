import json
import re

try:
    from llm.gemini import GeminiLLM
    from llm.prompts import PROMPT, build_prompt
except ImportError:
    from graphRAG.llm.gemini import GeminiLLM
    from graphRAG.llm.prompts import PROMPT, build_prompt


class AnswerGenerator:

    def __init__(self, llm=None):
        self.llm = llm or GeminiLLM().get_llm()

    def generate(self, question: str, contexts, history=None) -> dict:
        is_follow_up = bool(history)
        context = "\n\n".join(
            getattr(item, "text", str(item)) for item in contexts
        )

        prompt = build_prompt(
            context=context,
            question=question,
        )

        history_text = "\n".join(
            f"{item.get('role', 'user')}: {item.get('content', '')}"
            for item in (history or [])[-8:]
        )

        response_instruction = """

ĐÂY LÀ CÂU HỎI ĐÀO SÂU TRONG CUỘC HỘI THOẠI.
- Trả lời trực tiếp đúng câu hỏi mới, không giới thiệu lại ngành.
- Không tạo các mục "Tổng quan", "Vì sao phù hợp" hoặc "Bạn có thể làm gì tiếp theo".
- Có thể dùng Markdown ngắn gọn như danh sách, chữ đậm nếu giúp câu trả lời dễ đọc.
- Chỉ trả về JSON: {"noi_dung_tra_loi": "Câu trả lời trực tiếp bằng Markdown"}
""" if is_follow_up else ""

        full_prompt = (
            PROMPT
            + "\n\nLỊCH SỬ HỘI THOẠI:\n"
            + (history_text or "Chưa có")
            + response_instruction
            + "\n\n"
            + prompt
        )

        raw_answer = self.llm.complete(full_prompt).text.strip()

        # Một số LLM vẫn bọc JSON trong ```json ... ``` dù prompt đã yêu cầu.
        cleaned_answer = re.sub(
            r"^```(?:json)?\s*|\s*```$",
            "",
            raw_answer,
            flags=re.IGNORECASE,
        ).strip()

        try:
            answer = json.loads(cleaned_answer)
            if not isinstance(answer, dict):
                raise ValueError("LLM response is not a JSON object")
        except (json.JSONDecodeError, ValueError):
            # Không làm giao diện trống nếu model trả về văn bản thường.
            suggested_major = question.splitlines()[0].strip()
            answer = {
                "ten_nganh": suggested_major,
                "mo_ta_nganh": raw_answer,
                "ly_do_phu_hop": "Nội dung tư vấn được trình bày ở phần mô tả.",
                "phu_hop_nhom": True,
                "thong_bao_dinh_huong": "",
                "goi_y_tiep_theo": "Bạn có thể hỏi thêm về chương trình học, kỹ năng cần có hoặc cơ hội nghề nghiệp.",
                "nguon_tham_khao": "",
                "noi_dung_tra_loi": raw_answer,
            }

        if is_follow_up:
            direct_answer = str(
                answer.get("noi_dung_tra_loi")
                or answer.get("mo_ta_nganh")
                or raw_answer
            )
            return {
                "loai_phan_hoi": "tra_loi_tiep",
                "noi_dung_tra_loi": direct_answer,
            }

        group_fit = answer.get("phu_hop_nhom", True)
        if isinstance(group_fit, str):
            group_fit = group_fit.strip().lower() not in {"false", "0", "không", "khong"}

        return {
            "loai_phan_hoi": "tu_van_ban_dau",
            "ten_nganh": str(answer.get("ten_nganh", "")),
            "mo_ta_nganh": str(answer.get("mo_ta_nganh", "")),
            "ly_do_phu_hop": str(answer.get("ly_do_phu_hop", "")),
            "phu_hop_nhom": bool(group_fit),
            "thong_bao_dinh_huong": str(answer.get("thong_bao_dinh_huong", "")),
            "goi_y_tiep_theo": str(answer.get("goi_y_tiep_theo", "")),
            "nguon_tham_khao": str(answer.get("nguon_tham_khao", "")),
        }
