import openai 
import pinecone


def get_embeddings(text_to_embed):
    response = openai.Embedding.create(
    	model= "text-embedding-ada-002",
    	input=[text_to_embed]
	)
	# Extract the AI output embedding as a list of floats
    embedding = response["data"][0]["embedding"]
    return embedding


def initialize_pinecone(pinecone_api_key, pinecone_env, index="recipe-search"):
            pinecone.init(
                api_key=pinecone_api_key,
                environment=pinecone_env
                )

            recipe_index = pinecone.Index(index)
            return recipe_index

def generate_query(ingres, cuisine_types, expected_calorie, other_text):
    query = f"""
    Here are my ingredients: {ingres}. 
    My favorite cuisine types are {cuisine_types}. 
    I would like to control my calorie inputs of this meal between {expected_calorie[0]} and {expected_calorie[1]}.
    Here are some of my other requests: {other_text}.
    """
    return query