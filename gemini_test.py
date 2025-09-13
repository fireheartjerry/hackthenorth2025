from google import genai
import json

# Configure with your Gemini API key
client = genai.Client(api_key='AIzaSyBxiHLZX600rK0T-t8OHf_lw0d9a6qRwXo')

with open('data.json', 'r') as file:
    data = json.load(file)

appetite = """- Accept if product_type = "Commercial Property" AND state = "NY"
- Reject if premium < $50,000
- Prioritize if submission_date within last 7 days"""

appetites = []

for item in data["output"][0]["data"]:

    prompt = """You are an insurance agent that needs to classify insurance 'submissions' into one of the following 
    categories: 'Low Appetite', 'Medium Appetite', 'High Appetite' ONLY SAY THE CATEGORY AND NOTHING ELSE AND NO newline based on the following appetite criteria: """

    # Example submission
    submission = item

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt + appetite + " submission data: " + str(submission)
    )

    # print(response)
    appetites.append(response.candidates[0].content.parts[0].text)

print(appetites)

with open("output.json", 'w') as json_file:
    json.dump(appetites,"output.json",indent=4)



