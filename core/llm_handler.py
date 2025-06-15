from langchain_community.chat_models import ChatOllama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_cohere import CohereRerank
from langchain.retrievers import ContextualCompressionRetriever
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from config import ollama_model_name, COHERE_API_KEY, cohere_reranking_model_name, device, GOOGLE_API_KEY
from google.generativeai.types import HarmCategory, HarmBlockThreshold
# @st.cache_resource(show_spinner="Đang khởi tạo mô hình AI...")
# def get_llm_instance():
#     """Khởi tạo LLM instance."""
#     try:
#         print("[llm_handler] Bắt đầu khởi tạo ChatOllama...")
#         return ChatOllama(
#             model=ollama_model_name,
#             temperature=0.2,
#             device=device
#         )
#     except Exception as e:
#         print(f"[llm_handler] Lỗi khi khởi tạo LLM: {e}")
#         st.error(f"Lỗi khi khởi tạo LLM: {e}")
#         return None

@st.cache_resource(show_spinner="Đang khởi tạo mô hình AI (Gemini)...")
def get_llm_instance():
    """Khởi tạo LLM instance từ Google (Gemini) với định dạng safety_settings đúng."""
    from config import GOOGLE_API_KEY
    
    if not GOOGLE_API_KEY:
        st.error("Vui lòng thêm GOOGLE_API_KEY vào file config.py.")
        return None
        
    try:
        print("[llm_handler] Bắt đầu khởi tạo ChatGoogleGenerativeAI (Gemini)...")
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

        return ChatGoogleGenerativeAI(
            model="gemini-1.5-flash-latest",
            google_api_key=GOOGLE_API_KEY,
            temperature=0.1,
            convert_system_message_to_human=True,
            safety_settings=safety_settings
        )
    except Exception as e:
        print(f"[llm_handler] Lỗi khi khởi tạo Gemini: {e}")
        st.error(f"Lỗi khi khởi tạo mô hình Gemini: {e}")
        return None

@st.cache_resource(show_spinner="Đang khởi tạo mô hình Reranker...")
def get_reranker():
    """Khởi tạo Cohere Reranker."""
    if not COHERE_API_KEY:
        print("[llm_handler] COHERE_API_KEY không được cung cấp. Không thể khởi tạo reranker.")
        return None
    try:
        print("[llm_handler] Bắt đầu khởi tạo CohereRerank...")
        return CohereRerank(
            cohere_api_key=COHERE_API_KEY,
            model=cohere_reranking_model_name,
            top_n=5
        )
    except Exception as e:
        print(f"[llm_handler] Lỗi khi khởi tạo Cohere Reranker: {e}")
        st.error(f"Lỗi khi khởi tạo Cohere Reranker: {e}")
        return None

def reorder_documents(docs):
    """
    Sắp xếp lại tài liệu để tăng hiệu quả của LLM bằng cách đặt một số tài liệu ưu tiên lên đầu.
    Thông thường, LLM đọc và nhớ tốt hơn những nội dung ở đầu và cuối context.
    """
    print(f"[llm_handler] Đang sắp xếp lại {len(docs)} tài liệu tham khảo để tối ưu...")
    if not docs or len(docs) == 0:
        return ""
    
    reordered_docs = []
    if len(docs) > 1:
        reordered_docs.append(docs[0])
        
        if len(docs) > 2:
            reordered_docs.extend(docs[2:])
        
        if len(docs) > 1:
            reordered_docs.append(docs[1]) 
    else:
        reordered_docs = docs
    
    context_text = "\n\n---\n\n".join([doc.page_content for doc in reordered_docs])
    return context_text

def get_qa_retrieval_chain(llm_instance, retriever):
    """Khởi tạo RetrievalQA chain với LLM và retriever đã có."""
    if not llm_instance or not retriever:
        print("[llm_handler] LLM instance hoặc retriever không hợp lệ.")
        st.error("LLM instance hoặc retriever không hợp lệ.")
        return None
    
    print("[llm_handler] Bắt đầu khởi tạo RetrievalQA chain...")
    
    prompt_template_str = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Bạn là một trợ lý AI chuyên nghiệp, chỉ được phép trả lời dựa trên "Thông tin tham khảo" bên dưới. Nếu không tìm thấy thông tin, hãy trả lời đúng là "Thông tin này không có trong tài liệu được cung cấp.".

**Quy trình làm việc:**
1. Đọc kỹ câu hỏi và toàn bộ "Thông tin tham khảo" được cung cấp.
2. Tìm kiếm, xác định và trích xuất các đoạn, câu, dữ kiện hoặc ý liên quan trực tiếp đến câu hỏi trong tài liệu.
3. Đối chiếu, tổng hợp, so sánh và phân tích các thông tin liên quan này để tạo thành một câu trả lời ngắn gọn, rõ ràng, đúng trọng tâm, có thể so sánh hoặc làm rõ nếu cần thiết.
4. Nếu có nhiều nguồn hoặc ý trong tài liệu, hãy tổng hợp, so sánh, hoặc làm rõ các điểm khác biệt hoặc bổ sung lẫn nhau.
5. Nếu không có thông tin phù hợp và không thể đưa ra câu trả lời, trả lời đúng: "Thông tin này không có trong tài liệu được cung cấp." (không thêm thắt gì khác).
6. Nếu chỉ tìm thấy một phần của câu hỏi trong dữ liệu, hãy cố đưa ra câu trả lời và chỉ rõ bạn tìm thấy phần nào, không thấy phần nào.
7. Luôn luôn trả lời bằng Tiếng Việt.

**Yêu cầu nghiêm ngặt:**
- Chỉ sử dụng thông tin có trong "Thông tin tham khảo".
- Không được bịa, không được tự chế, không được tự đưa ra nhận định ngoài tài liệu, không sử dụng kiến thức riêng của bạn.
- Không được nói lại yêu cầu, không được nói lại câu hỏi, không được nói lại nội dung tài liệu.
- Trả lời bằng Tiếng Việt một cách ngắn gọn, đúng trọng tâm, không giải thích lan man, không thêm thắt.

Thông tin tham khảo:
{context}<|eot_id|><|start_header_id|>user<|end_header_id|>
Câu hỏi: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
Câu trả lời hữu ích:"""

    try:
        PROMPT = PromptTemplate(
            template=prompt_template_str, input_variables=["context", "question"]
        )
        
        reranker = get_reranker()
        if reranker:
            print("[llm_handler] Sử dụng Cohere Reranker để cải thiện kết quả...")
            try:
                compression_retriever = ContextualCompressionRetriever(
                    base_compressor=reranker,
                    base_retriever=retriever
                )
                retriever = compression_retriever
            except Exception as e:
                print(f"[llm_handler] Lỗi khi khởi tạo contextual compression: {e}")
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm_instance,
            chain_type="stuff", 
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        
        print("[llm_handler] Đã khởi tạo xong RetrievalQA chain.")
        return qa_chain
    except Exception as e:
        print(f"[llm_handler] Lỗi khi tạo QA chain: {e}")
        st.error(f"Lỗi khi tạo QA chain: {e}")
        return None