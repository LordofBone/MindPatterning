# From https://github.com/alexandrebarachant/muse-lsl/blob/master/examples/startMuseStream.py
from muselsl import stream, list_muses, muse


def start_stream():
    muses = list_muses()

    if not muses:
        print('No Muses found')
    else:
        try:
            stream(str(muses[0]['address']))

            # Note: Streaming is synchronous, so code here will not execute until the stream has been closed
        except(KeyboardInterrupt):
            print('Stream has ended')


if __name__ == "__main__":
    start_stream()
