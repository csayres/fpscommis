import asyncio
import time


nIter = 70
metLevel = 1
expTime = 1
danger = True
collisionBuffer = 2


async def sendJaegerCommand(cmdStr):
    cmdID = 3

    reader, writer = await asyncio.open_connection(
        'localhost', 19990)

    while True:
        data = await reader.readline()
        data = data.decode()
        if "yourUserID=" in data:
            userID = int(data.split("yourUserID=")[-1].split(";")[0])
            print("myUserID %i"%userID)
        if "version=" in data:
            # print("break!")
            break

    cmdStr = "%i %s\n"%(cmdID, cmdStr)

    print(f'Send: %s'%cmdStr)
    writer.write(cmdStr.encode())
    await writer.drain()
    while True:
        data = await reader.readline()
        data = data.decode()
        print("read from fvc:", data)
        if "%i %i :"%(userID, cmdID) in data:
            print("command succeeded")
            success = True
            break
        if "%i %i f"%(userID, cmdID) in data:
            print("command failed")
            success = False
            break

    print('Close the connection')
    writer.close()
    await writer.wait_closed()

    return success


async def main():
    for ii in range(nIter):
        tstart = time.time()
        print("iter: %i\n\n"%ii)
        configCmd = "configuration random"
        if danger:
            configCmd += " --danger --max-retries=10 --collision-buffer=%.2f"%collisionBuffer
        success = await sendJaegerCommand(configCmd)
        print("configuration random success", success)
        if success==False:
            print("configuration random failed, exiting")
            return
        success = await sendJaegerCommand("fvc loop --no-apply --fbi-level %.2f --exposure-time %.2f"%(metLevel, expTime))
        print("fvc image complete")
        success = await sendJaegerCommand("configuration reverse")
        print("configuration reverse success", success)
        if success==False:
            print("configuration reverse failed, exiting")
            return
        print("iter took %.2f mins"%((time.time()-tstart)/60.))

if __name__ == "__main__":
    asyncio.run(main())
