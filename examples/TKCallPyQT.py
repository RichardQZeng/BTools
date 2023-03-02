import tkinter as tk
import subprocess

window = tk.Tk()
window.title("Window Title")
window.geometry('200x100')

lbl = tk.Label(window, text="Hello World")
lbl.pack()

def clicked(): #function before bind
    subprocess.Popen(['python' , 'BatchProcessing.py'])

btn = tk.Button(window, text="Click Me", command=clicked)
btn.pack()

window.mainloop()