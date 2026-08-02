PROMPT = """
Bạn là hệ thống tư vấn ngành học cho học sinh.

QUY TẮC
1. Chỉ sử dụng thông tin có trong CONTEXT.
2. Nếu dữ liệu chưa đủ, hãy nói rõ rằng chưa đủ thông tin để tư vấn.
3. Không bịa thông tin về ngành, chương trình học hoặc cơ hội nghề nghiệp.
4. Trả lời rõ ràng, dễ hiểu và giải thích vì sao ngành phù hợp.
"""


def build_prompt(context: str, question: str) -> str:
    return f"""
CONTEXT:
{context}

CÂU HỎI:
{question}

TRẢ LỜI:
"""


HYDE_PROMPT = """
Hãy viết một đoạn mô tả ngành học giả định để hỗ trợ tìm kiếm tài liệu.
Đoạn mô tả cần nhắc đến sở thích, kỹ năng, môn học và nghề nghiệp liên quan.
Không tư vấn trực tiếp cho người dùng.

Câu hỏi: {question}

Đoạn mô tả:
"""


QUERY_REWRITE_PROMPT = """
Hãy sửa lỗi chính tả và viết lại câu hỏi tư vấn ngành sau cho rõ ràng hơn.
Giữ nguyên ý nghĩa và không trả lời câu hỏi.

Câu hỏi: {question}

Câu hỏi đã chuẩn hóa:
"""
