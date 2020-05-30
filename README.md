# 20192 machine learning and datamining
  - **Save code topic classification - vn news**
  - link data: [link](https://github.com/trongtuyen99/vietnamese_news_ml_dl)

# Hệ thống cho phép phân loại theo chủ đề của văn bản đầu vào:
1. Cài đặt môi trường
    - python: >=3.6
    - cài đặt các thư viện cần thiết:
        - ```pip install scikit-learn numpy matplotlib pyvi nltk gensim joblib```
        - nếu chạy lần đầu, download thêm data punkt của thư viện nltk: 
            -   ```python
                import nltk
                nltk.download('punkt')
                ```
                
        - nếu muốn chạy file demo_web: install thêm streamlit:
            - ```pip install streamlit```
            
2. Chạy demo
- demo trên console:
    - cd đến thư mục demo rồi chạy file demo_console.py
- demo trên web:
    - cd đến thư mục demo, chạy command line
        ```streamlit run demo_web.py``` 
