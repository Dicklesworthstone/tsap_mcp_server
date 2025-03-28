from utils import process_data

# TODO: Add error handling
API_KEY = "HARDCODED_SECRET_abc123xyz789" # Example secret

def main_function(input_path):
    # This might be vulnerable to SQL injection if input_path is user-controlled
    query = f"SELECT * FROM data WHERE path = '{input_path}'"
    print(f"Executing: {query}")
    data = load_data(input_path)
    processed = process_data(data)  # noqa: F841
    print("Done")

def load_data(path):
    # Insecure file loading
    with open(path, 'r') as f:
        return f.read()

if __name__ == "__main__":
    main_function("data.txt")