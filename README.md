---
title: "RAG"
emoji: "🐀"
colorFrom: "red"
colorTo: "blue"
sdk: "streamlit"
sdk_version: "1.32.2"
app_file: app.py
pinned: false
---

[doc](https://docs.google.com/document/d/1JgFa-yZ-Bz_SyAq-5Hys2kKD5B4D8uK31oWhGEShrMw/edit?usp=sharing)
[pptx](https://docs.google.com/presentation/d/1Vx7SjbXEJYPe7cF_0xskG_QhVXhlF6dgDjGdkOPRv5M/edit?usp=sharing)
# Rất Ảo Giác - Vietnamese RAG Chatbot

Dự án này là thành quả của nhóm trong bài tập lớn môn Deep Learning (2425II_AIT3001*_1). Nhóm xin gửi lời cảm ơn chân thành đến thầy Triệu Hải Long đã tận tình hướng dẫn, đồng hành với chúng em trong suốt quá trình thực hiện dự án.

Thành viên nhóm:

Lê Vũ Hiếu - 23020365

Đàm Lê Minh Quân - 23020416

Nguyễn Hoàng Tú - 23020428


Hệ thống RAG (Retrieval Augmented Generation) nâng cao cho tài liệu Tiếng Việt sử dụng Llama 3, LangChain và Streamlit.

## Giới thiệu về RAG

RAG (Retrieval Augmented Generation) là kỹ thuật kết hợp tìm kiếm thông tin (retrieval) và tạo sinh văn bản (generation) để cung cấp câu trả lời chính xác dựa trên tài liệu. Thay vì LLM phải "tưởng tượng" thông tin, nó được cung cấp thông tin liên quan từ tài liệu nguồn, giúp:

- **Tăng độ chính xác**: Giảm thiểu hiện tượng "ảo tưởng" (hallucination)
- **Nguồn có thể xác minh**: Trích dẫn thông tin từ tài liệu cụ thể
- **Cập nhật kiến thức**: Không bị giới hạn bởi dữ liệu huấn luyện
- **Cá nhân hóa**: Tích hợp và trả lời dựa trên dữ liệu riêng của người dùng

## Tính năng nâng cao

Ứng dụng này sử dụng các kỹ thuật RAG nâng cao:

1. **Parent-Child Document Chunking**: Tạo các chunk cha lớn (để truy xuất và hiểu ngữ cảnh rộng) và chunk con nhỏ (để tìm kiếm chính xác)
2. **Hybrid Search**: Kết hợp BM25 (từ khóa) và Embedding Search (ngữ nghĩa) để cải thiện kết quả tìm kiếm 
3. **Cohere Reranking**: Sử dụng AI để sắp xếp lại kết quả tìm kiếm theo thứ tự phù hợp nhất
4. **Document Reordering**: Sắp xếp lại tài liệu đầu vào để tối ưu hiệu suất của LLM
5. **Advanced Prompting**: Cấu trúc prompt tối ưu hướng dẫn LLM phân tích và trả lời

## Cài đặt và chạy

### Yêu cầu

- Python 3.9+
- [Ollama](https://ollama.ai/) (để chạy mô hình Llama3 cục bộ)

### Các bước cài đặt

1. Cài đặt dependencies:
```bash
pip install -r requirements.txt
```

2. Tải mô hình Llama3 bằng Ollama:
```bash
ollama pull llama3:8b-instruct-q4_0
```

3. Chạy ứng dụng:
```bash
streamlit run app.py
```

### Sử dụng

1. Tải lên tài liệu văn bản (.txt, .pdf, .doc, .docx)
2. Đợi hệ thống xử lý tài liệu
3. Đặt câu hỏi liên quan đến nội dung tài liệu
4. Xem câu trả lời và nguồn tham khảo

## Cấu trúc dự án

```
DLFinal-RAG/
├── app.py                  # Ứng dụng Streamlit chính
├── config.py               # Cấu hình hệ thống
├── core/
│   ├── document_processor.py  # Xử lý và chia chunk tài liệu
│   ├── embedding_handler.py   # Tạo và quản lý vector embeddings
│   ├── llm_handler.py         # Xử lý LLM và chuỗi QA
│   ├── chat_history.py        # Quản lý lịch sử chat
├── ui/
│   ├── chat_interface.py      # Giao diện chat
│   ├── sidebar.py             # Sidebar cho ứng dụng
├── data/                   # Thư mục lưu trữ dữ liệu
│   ├── vector_stores/         # Vector stores
│   ├── chat_histories/        # Lịch sử chat
│   ├── uploaded_files/        # Files đã tải lên
├── requirements.txt        # Dependencies
├── README.md               # Tài liệu
```

## Tùy chỉnh

- Thay đổi mô hình nhúng: Chỉnh sửa `embedding_model_name` trong `config.py`
- Thay đổi mô hình LLM: Chỉnh sửa `ollama_model_name` trong `config.py`
- Thay đổi kích thước chunk: Điều chỉnh `PARENT_CHUNK_SIZE`, `CHILD_CHUNK_SIZE` trong `config.py`
- API key Cohere: Cập nhật `COHERE_API_KEY` trong `config.py` để sử dụng dịch vụ reranking

## Lưu ý

- Ứng dụng yêu cầu Ollama chạy trên máy cục bộ để hoạt động
- Cần đủ RAM để xử lý tài liệu lớn và chạy mô hình Llama3
- Để đạt hiệu quả tốt nhất, các tài liệu nên có định dạng rõ ràng và ngôn ngữ Tiếng Việt
