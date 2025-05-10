import streamlit as st
import base64

st.set_page_config(page_title="HOME", page_icon="",layout="wide")
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

image1 = get_base64_image("images/menu01.png")
image2 = get_base64_image("images/menu02.png")
image3 = get_base64_image("images/menu03.png")
image4 = get_base64_image("images/menu04.png")

#st.title("HOME")
# カスタムCSSを適用
st.markdown(
    """
    <style>
    .card {
        /* カードの基本スタイル */
        background-color: #fff;
        border: 1px solid rgba(0, 0, 0, 0.125);
        border-radius: 0.25rem;
        margin-bottom: 1.5rem;
        overflow: hidden;
    }
    .card-img-top {
        /* カード上部の画像 */
        width: 100%;
        height: auto;
        object-fit: cover;
    }
    .card-body {
        /* カードのコンテンツ部分 */
        padding: 1.25rem;
        text-align: center;
    }
    .card-title {
        /* タイトル */
        font-size: 1.25rem;
        font-weight: 500;
        margin-top: 0;
        margin-bottom: 0.75rem;
    }
    .card-text {
        /* 本文 */
        font-size: 1rem;
        color: #333;
        margin-bottom: 1rem;
    }
    .card-button {
        /* ボタン */
        display: inline-block;
        margin: auto;
        padding: 0.5rem 1rem;
        font-size: 1rem;
        color: #333;
        background-color: #fdf0e2;
        border: none;
        border-radius: 0.25rem;
        text-decoration: none;
        cursor: pointer;
    }
    .card-button:hover {
        background-color: #fad2b1;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.image("images/hero.jpg", use_container_width=True)

col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
with col1:
    st.markdown(
        f"""
        <div class="card">
            <img src="data:image/png;base64,{image1}" class="card-img-top" alt="音声推薦">
            <div class="card-body">
                <h5 class="card-title">音声推薦</h5>
                <p class="card-text">ページ解説</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        f"""
        <div class="card">
            <img src="data:image/png;base64,{image2}" class="card-img-top" alt="ボクセルアート">
            <div class="card-body">
                <h5 class="card-title">３Dボクセルアート</h5>
                <p class="card-text">ページ解説</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        f"""
        <div class="card">
            <img src="data:image/png;base64,{image3}" class="card-img-top" alt="音声一覧">
            <div class="card-body">
                <h5 class="card-title">サンプル音声一覧</h5>
                <p class="card-text">ページ解説</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col4:
    st.markdown(
        f"""
        <div class="card">
            <img src="data:image/png;base64,{image4}" class="card-img-top" alt="音声クラスタリング">
            <div class="card-body">
                <h5 class="card-title">音声クラスタリング</h5>
                <p class="card-text">ページ解説</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
