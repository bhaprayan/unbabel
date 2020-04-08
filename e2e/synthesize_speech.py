import os
import sys
import threading
from contextlib import closing

import pandas as pd
from boto3 import Session
from botocore.exceptions import BotoCoreError, ClientError

# Create a client using the credentials and region defined in the [adminuser]
# section of the AWS credentials file (~/.aws/credentials).


def synthesize(text, accent="us"):
    session = Session(profile_name="default")
    polly = session.client("polly")
    accent = accent.lower()
    if accent == "us":
        voice_id = "Joey"
    elif accent == "uk":
        voice_id = "Brian"
    else:
        voice_id = "Joey"

    try:
        # Request speech synthesis
        response = polly.synthesize_speech(
            Text=text, OutputFormat="mp3", VoiceId=voice_id,
        )
    except (BotoCoreError, ClientError) as error:
        # The service returned an error, exit gracefully
        print(error)
        sys.exit(-1)

    # Access the audio stream from the response
    if "AudioStream" in response:
        # Note: Closing the stream is important because the service throttles on the
        # number of parallel connections. Here we are using contextlib.closing to
        # ensure the close method of the stream object will be called automatically
        # at the end of the with statement's scope.
        with closing(response["AudioStream"]) as stream:
            fn = text + "_" + accent + ".mp3"
            output = os.path.join(os.getcwd(), fn)

            try:
                # Open a file for writing the output as a binary stream
                with open(output, "wb") as file:
                    file.write(stream.read())
            except IOError as error:
                # Could not write to file, exit gracefully
                print(error)
                sys.exit(-1)

    else:
        # The response didn't contain audio data, exit gracefully
        print("Could not stream audio")
        sys.exit(-1)


if __name__ == "__main__":
    tokens = pd.read_csv("words.csv")["words"].to_list()
    print("Synthesizing number of words:", len(tokens))
    for token in tokens:
        accent = "uk"
        t = threading.Thread(target=synthesize, args=(token, accent)).start()
