# import streamlit as st
from st_pages import Page, Section, show_pages, add_page_title

# st.set_page_config(page_title="Intelli Coin Trader", page_icon="📊")



show_pages(
    [
        Page("pages/welcome.py", "Home", "🏠"),
        Page("pages/EDA.py", "EDA", "📈"),
        Page("pages/Model_Evaluations.py", "Model Evaluations", "📈"),
        # Page("other_pag/page2.py", "Page 2", ":books:"),
        Section("Forecaster", icon="🎈️"),
        Page("pages/Moving_Average.py", "Moving Average", "📈"),
        Page("pages/Trade_Assistant.py", "Trade Assistant", "📈"),
        # Pages after a section will be indented
        # Page("Another page", icon="💪"),
        # Unless you explicitly say in_section=False
        # Page("Not in a section", in_section=False)
    ]
)

add_page_title()


# with st.echo("below"):
#     from st_pages import Page, Section, add_page_title, show_pages

#     "## Declaring the pages in your app:"

#     show_pages(
#       [
#         Page("pages/welcome.py", "Home", "🏠"),
#         Page("pages/2_📈_EDA.py", "EDA", "📈"),
#         Page("pages/3_📈_Model_Evaluations.py", "Model Evaluations", "📈"),
#         # Page("other_pages/page2.py", "Page 2", ":books:"),
#         Section("Forecast", icon="🎈️"),
#         Page("pages/4_📈_Moving_Average.py", "Moving Average", "📈", in_section=True),
#         Page("pages/5_📈_Trade_Assistant.py", "Trade Assistant", "📈", in_section=True),
#         # Pages after a section will be indented
#         # Page("Another page", icon="💪"),
#         # Unless you explicitly say in_section=False
#         # Page("Not in a section", in_section=False)
#       ]
#     )

#     add_page_title()  # Optional method to add title and icon to current page

# def intro():
#     import streamlit as st

#     st.write("# Welcome to Streamlit! 👋")
#     st.sidebar.success("Select a demo above.")

#     st.markdown(
#         """
#         Streamlit is an open-source app framework built specifically for
#         Machine Learning and Data Science projects.

#         **👈 Select a demo from the dropdown on the left** to see some examples
#         of what Streamlit can do!

#         ### Want to learn more?

#         - Check out [streamlit.io](https://streamlit.io)
#         - Jump into our [documentation](https://docs.streamlit.io)
#         - Ask a question in our [community
#           forums](https://discuss.streamlit.io)

#         ### See more complex demos

#         - Use a neural net to [analyze the Udacity Self-driving Car Image
#           Dataset](https://github.com/streamlit/demo-self-driving)
#         - Explore a [New York City rideshare dataset](https://github.com/streamlit/demo-uber-nyc-pickups)
#     """
#     )

# page_names_to_funcs = {
#     "—": intro,
#     # "Mapping Demo": mapping_demo
# }

# demo_name = st.sidebar.selectbox("Choose a demo", page_names_to_funcs.keys())
# demo_name = st.sidebar()
# page_names_to_funcs[demo_name]()

