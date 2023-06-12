import uvicorn, sys, time, os
from fastapi import FastAPI, Request, Form
import logging
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from app.models import predictModel
# logging = logger()

app = FastAPI()


@app.get("/health")
async def health():
    return {"status": "up"}

@app.post("/get_answer")
async def get_answer(data: Request):
    """
  Fastapi POST method that gets the best question and answer
  in the set context.

  Args:
    data(`dict`): One field required 'questions' (`list` of `str`)

  Returns:
    A `dict` containing the original question ('orig_q'), the most similar
    question in the context ('best_q') and the associated answer ('best_a').
  """
    t1 = time.time()
    data = await data.json()
    text_input = data["questions"]
    response = predictModel.prediction_model(text_input)
    logging.info('result：{}'.format(response))
    t2 = time.time()
    print('耗时', t2-t1)
    return response

# initialises the QA model and starts the uvicorn app
if __name__ == "__main__":
    logging.info('start servering ....')
    logging.info('Initializing nlu model successfully and start nlu service')

    uvicorn.run(app, host="0.0.0.0", port=9011)

