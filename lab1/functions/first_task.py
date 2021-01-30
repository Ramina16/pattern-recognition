import websockets
import json
import numpy as np

x = {"data": {"message": "Let's start"}}


def pause():
    program_pause = input("Press the <ENTER> key to continue...")


def xor(num1, num2):
    """
    This function calculates a logical XOR of two numbers(0 or 1)
    Define input and expected output:
    >>> xor(0, 1)
    1
    >>> xor(1, 1)
    0
    >>> xor(-1, 1)
    Traceback (most recent call last):
        ...
    ValueError: numbers must be 1 or 0
    """
    if num1 == num2 == 0 or num1 == num2 == 1:
        return 0
    elif (num1 == 1 and num2 == 0) or (num1 == 0 and num2 == 1):
        return 1
    else:
        raise ValueError('numbers must be 1 or 0')


def first_task(y, y1, noise):
    """
    This function calculates the number with the highest probability
    Args:
        y: dict with np.arrays that match the etalons
        y1: list with np.array of number with some noise
        noise: float between 0 and 1
    Define input and expected output:
    >>> first_task({'0': [[0, 0], [1, 1]], '1': [[1, 0], [0, 1]]},[[0, 1], [1, 0]], 1)
    1
    >>> first_task({'0': [[0, 0, 0]], '1': [[0, 1, 0]]}, [[1, 0, 0]], 0.4)
    0
    >>> first_task({'0': [[0, 0, 0], [1, 1, 1]], '1': [[0, 1, 0], [1, 0, 0]]}, [[1, 0, 0], [0, 0, 1]], 0.4)
    0
    """
    y1 = np.array(y1)
    k_list = [key for key in y]
    argmax = []
    xor_list = [[] for k in range(len(y))]
    # filling in conditional probabilities
    k = 0
    while k < len(k_list):
        for number in k_list:
            for i in range(len(y1)):
                for j in range(len(y1[i])):
                    xor_ans = xor(y1[i][j], np.array(y[number][i][j]))
                    xor_list[k].append(xor_ans*noise + xor(1, xor_ans)*(1-noise))
            k += 1
    # filling in sums of probabilities
    for i in range(len(xor_list)):
        argmax.append(sum(xor_list[i]))
    print(argmax)
    for i in range(len(argmax)):
        argmax[i] = abs(argmax[i])
    res = np.argmax(argmax)
    return res


async def first():
    """
    This function connect to the websocket server, send messages and get data from the server.
    Also it checks scales of values
    """
    async with websockets.connect('wss://sprs.herokuapp.com/first/helen', max_size=1000000000) as websocket:
        await websocket.send(json.dumps(x))
        response = await websocket.recv()
        print(response)
        pause()  # pause for login(if we want) to wss://sprs.herokuapp.com/first/...
        width = int(input('width: '))
        if width > 100 or width <= 0:  # check width
            print('width must be between 1 and 100')
            width = int(input('width: '))
        height = int(input('height: '))
        if height > 100 or height <= 0:  # check height
            print('height must be between 1 and 100')
            height = int(input('height: '))
        total_steps = int(input('total_steps: '))
        if total_steps > 1000000 or total_steps <= 0:  # check count of steps
            print('total_steps must be between 1 and 1 000 000')
            total_steps = int(input('total_steps: '))
        noise = float(input('noise: '))
        if noise > 1 or noise < 0:  # check noise
            print('noise must be between 0 and 1')
            noise = float(input('noise: '))
        j = {"data": {"width": width, "height": height, "totalSteps": total_steps, "noise": noise, "shuffle": False}}
        await websocket.send(json.dumps(j))
        response1 = await websocket.recv()
        y = json.loads(response1)  # correct numbers
        for i in range(total_steps):
            k = {"data": {"message": "Ready"}}
            await websocket.send(json.dumps(k))
            response2 = await websocket.recv()
            y1 = json.loads(response2)  # numbers with noise
            result = first_task(y['data'], y1['data']['matrix'], noise)
            print('result:', result)
            m = {"data": {"step": i+1, "answer": str(result)}}
            await websocket.send(json.dumps(m))
            response3 = await websocket.recv()
            print(response3)
            pause()
        bye = {"data": {"message": "Bye"}}
        await websocket.send(json.dumps(bye))
        response4 = await websocket.recv()
        print(response4)

if __name__ == "__main__":
    import doctest
    doctest.testmod()
