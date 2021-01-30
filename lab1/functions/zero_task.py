import asyncio
import websockets
import json


def zero_task(a, b, operator):
    if operator == '+':
        res = a + b
        return res
    elif operator == '-':
        res = a - b
        return res
    elif operator == '*':
        res = a * b
        return res
    else:
        print('error')


x = {"data": {"message": "Let's start"}}


async def zero():
    async with websockets.connect('wss://sprs.herokuapp.com/zeroth/helen') as websocket:
        await websocket.send(json.dumps(x))
        response = await websocket.recv()
        print(response)
        j = json.loads(response)
        a = j["data"]["operands"][0]
        b = j["data"]["operands"][1]
        operator = j["data"]["operator"]
        result = zero_task(a, b, operator)
        print('result:', result)
        y = {"data": {"answer": result}}
        await websocket.send(json.dumps(y))
        response1 = await websocket.recv()
        print(response1)

asyncio.get_event_loop().run_until_complete(zero())
