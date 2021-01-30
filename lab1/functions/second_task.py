import websockets
import json
import numpy as np


def pause():
    program_pause = input("Press the <ENTER> key to continue...")


def second_task(y, repeats):
    """
    This function calculates the coordinate of the best position to send an aim
    Args:
         y: list of positive integers not greater than 255, and representing the heatmap without normalization
         repeats: number attempts per one heatmap(number of people)
    Define input and expected output:
    >>> second_task([110, 100, 1], 2)
    [0, 0]
    >>> second_task([255, 200, 255, 155], 3)
    [1, 1, 1]
    """
    y = np.array(y)
    sum_y = y.sum()
    norm_heatmap = [i/sum_y for i in y]
    #print(norm_heatmap)
    sum_prob = 0
    guesses = []
    for i in range(len(norm_heatmap)):
        sum_prob += norm_heatmap[i]
        if sum_prob >= 1/2:
            for j in range(repeats):
                guesses.append(i)
            break
    return guesses


async def second():
    async with websockets.connect('wss://sprs.herokuapp.com/second/helen') as websocket:  # connecting
        width = int(input('width: '))
        if width > 100 or width < 2:  # check width
            print('width must be between 2 and 100')
            width = int(input('width: '))
        total_steps = int(input('total_steps: '))
        if total_steps > 1000000 or total_steps < 1:  # check count of steps
            print('total_steps must be between 1 and 1 000 000')
            total_steps = int(input('total_steps: '))
        repeats = int(input('repeats: '))
        if repeats > 1000 or repeats < 1:  # check count of repeats
            print('repeats must be between 1 and 1 000')
            repeats = int(input('repeats: '))
        x = {"data": {"width": width, "loss": "L1", "totalSteps": total_steps, "repeats": repeats}}
        await websocket.send(json.dumps(x))
        response = await websocket.recv()
        print(response)
        pause()  # pause for login(if we want) to wss://sprs.herokuapp.com/second/...
        for i in range(total_steps):
            k = {"data": {"message": "Ready"}}
            await websocket.send(json.dumps(k))
            response1 = await websocket.recv()
            print(response1)
            y = json.loads(response1)
            guesses = second_task(y['data']['heatmap'], repeats)
            m = {"data": {"step": i+1, "guesses": guesses}}
            await websocket.send(json.dumps(m))
            response2 = await websocket.recv()
            print(response2)
            pause()
        bye = {"data": {"message": "Bye"}}
        await websocket.send(json.dumps(bye))
        response4 = await websocket.recv()
        print(response4)

if __name__ == "__main__":
    import doctest
    doctest.testmod()
