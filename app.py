from flask import Flask, request, send_file 
from apis import generate_speech 

app = Flask(__name__)

@app.route('/')
def hello():
    return 'Hello, from tts!'

@app.route('/tts', methods=['POST'])
def tts():
    try:
        # Get data from the JSON payload of the POST request
        data = request.get_json()

        # Extract text and person from the request data
        text = data.get('text')
        person = data.get('person')

        # Call generate_speech function
        result_person = generate_speech(text, person)

        # Send the generated file back in the response
        return send_file(result_person, as_attachment=True)

    except Exception as e:
        # Handle exceptions and return an error response
        print(f"Error: {e}")
        return {"error": "Internal Server Error"}, 500


if __name__ == '__main__':
    app.run(debug=True)
