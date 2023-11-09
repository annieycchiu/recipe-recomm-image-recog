# Standard library imports
import os

# Third-party library imports
from dotenv import load_dotenv
import streamlit as st
import openai

# Custom module imports
from utils.helper import get_embeddings, initialize_pinecone, generate_query


def main():
    st.set_page_config(
        page_title="Let's Cook!",
        page_icon=":shallow_pan_of_food:",
        layout="wide",
    )

    st.header("Recipe Recommendation APP with Food Image Recognition")

    with st.sidebar:
        cuisine_types = st.text_input(
            "What's your favorite cuisine types?",
            placeholder="Japanese, Thai...")
        
        expected_calorie = st.slider(
            "Expected calorie inpts per meal", 
            300, 1200, (500, 800))
        
        other_text = st.text_input(
            "Do you have any other request?",
            placeholder="e.g. I am allergic to peanut...")
        
        ingres = st.text_input(
            "Ingredients:",
            placeholder="tomato, beef...")
        
        query = generate_query(ingres, cuisine_types, expected_calorie, other_text)
        st.write(query)


    if st.sidebar.button("Recommend recipes!"):
        N = 3
        query_embedding = get_embeddings(query)

        # Run the Query Search
        recipe_index = initialize_pinecone(pinecone_api_key, pinecone_env)
        outputs = recipe_index.query(query_embedding, top_k=N, include_metadata=True)['matches']

        for output in outputs:
            st.subheader(f'{output["metadata"]["recipe_name"]}')
            col1, col2 = st.columns([0.3, 0.7])
            
            st.write(f'Description: {output["metadata"]["description"]}')
            st.write(f'Prep time: {output["metadata"]["prep_time"]}')
            st.write(f'Cook time: {output["metadata"]["cook_time"]}')
            st.write(f'Servings: {output["metadata"]["servings"]}')

            st.divider()

if __name__ == "__main__":
    # Load environment variables from .env file
    load_dotenv()

    openai.api_key = os.getenv("OPENAI_API_KEY")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pinecone_env = os.getenv("PINECONE_ENVIRONMENT")

    main()


    