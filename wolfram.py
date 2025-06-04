import wolframalpha

client = wolframalpha.Client("666YV9-QAUUK34L67")

def query_wolfram(query):
    res = client.query(query)
    try:
        return next(res.results).text
    except StopIteration:
        return "No result from Wolfram."
