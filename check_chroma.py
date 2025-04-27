import chromadb

# Collection name
collection_name = "test"

# Establish a connection to the ChromaDB server
client = chromadb.HttpClient(
    host="ip-address",
    port=8000
)

# Create a collection
# from datetime import datetime
#
# collection = client.create_collection(
#     name=collection_name,
#     embedding_function=None,
#     metadata={
#         "description": collection_name + " collection",
#         "created": str(datetime.now())
#     }
# )

# Delete the collection if exists
# client.delete_collection(name=collection_name)

# Get the collection
collection = client.get_collection(name=collection_name)

# Check the collection
print(collection.peek())

# Count the number of documents in the collection
print(collection.count())
