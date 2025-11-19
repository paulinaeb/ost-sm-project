import streamlit as st

def add_footer(author_name):
    """
    Add a styled footer with author information to the page.
    
    Args:
        author_name (str): Name of the page author
    """
    st.markdown("""
        <style>
            .footer {
                position: fixed;
                bottom: 0;
                left: 0;
                width: 100%;
                background-color: rgba(14, 17, 23, 0.95);
                color: #fafafa;
                text-align: center;
                padding: 15px 0;
                font-size: 14px;
                border-top: 1px solid rgba(250, 250, 250, 0.1);
                z-index: 999;
            }
            .footer a {
                color: #00c6ff;
                text-decoration: none;
            }
            .footer a:hover {
                text-decoration: underline;
            }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown(
        f"""
        <div class="footer">
            👤 <strong>Author:</strong> {author_name} | 🇪🇺 Europe CyberScope — CSOMA Project
        </div>
        """,
        unsafe_allow_html=True
    )
