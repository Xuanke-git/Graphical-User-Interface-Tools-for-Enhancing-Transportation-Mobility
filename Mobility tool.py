import os
import tkinter as tk
from tkinter import messagebox, Toplevel
from PIL import Image, ImageTk
from LP_CAR_ind_Tn_41_h_11 import *
from geo_plotting import community


def open_main_window():
    # Close the splash screen
    splash_root.destroy()


def s1():
    print('Step 1 selected city:')
    selected_answer = var.get()
    print(choices[selected_answer], '\n')
    if selected_answer == 1:
        display_image("Charleston.png")
    elif selected_answer == 2:
        display_image("Clemson.png")
    elif selected_answer == 3:
        display_image("Columbia.png")
    elif selected_answer == 4:
        display_image("Greenville.png")
    elif selected_answer == 5:
        display_image("Myrtle Beach.png")
    else:
        messagebox.showerror("Error", "Please select an answer")


def s2():
    global arg
    print('Step 2 selected POIs:')
    for i in range(len(check_var)):
        if n := check_var[i].get():
            print(n, check_choices[n])
            arg.append(n)
    # display_image('POI_Columbia.png')
    community(choices[var.get()].rstrip(', SC'), arg)


def s3():
    parameters = dict(c=54, r=5, h_r=3, p=0, I_n=len(arg), S=90)
    #     - depreciation value c per vehicle per day
    #     - positive impact r, revenue per time period per car
    #     - reposition cost h_r, cost per time period per car
    #     - idle cost h_i, cost per time period per car
    #     - p penalty if demand is not fulfilled per request
    #     - budget B per day,
    #     - Zone number I_n
    #     - Time period T, time point T_n
    #     - Number of scenarios: S
    parameters['B'] = b_e.get()
    parameters['t'] = interval.get()
    parameters['p'] = p if len(p := penalty.get()) else 2
    parameters['h_i'] = i if (i := idle.get()) else 0.5
    print('\nStep 3 inputs:')
    print(f'Total budget: $ {float(b_e.get())}')
    print(f'Decision interval: {float(interval.get())} min')
    print(f'Minimum fulfillment: {float(fulfillment.get())} %')
    main(choices[var.get()].rstrip(', SC'), arg, parameters)


def display_image(image_path):
    try:
        new_window = Toplevel(root)
        new_window.title("Result output")
        # new_window.geometry("800x800")

        image = Image.open(image_path)
        # image = image.resize((300, 300), Image.ANTIALIAS)
        photo = ImageTk.PhotoImage(image)

        image_label = tk.Label(new_window, image=photo)
        image_label.image = photo  # Keep a reference to the image to prevent garbage collection
        image_label.pack()

    except IOError as e:
        messagebox.showerror("Error", f"Failed to load image: {e}")


def button_call(city, args, p):
    return main


splash_root = tk.Tk()
loading_image = Image.open('Loading.jpg')
loading_photo = ImageTk.PhotoImage(loading_image)
loading_label = tk.Label(splash_root, image=loading_photo)
loading_label.pack()
splash_root.geometry()
splash_root.after(9500, open_main_window)
splash_root.mainloop()
#
root = tk.Tk()
root.title("Mobility Analysis Tool")

n = 0
arg = []

# Choose the city
tk.Label(root, text="Step 1(*): Please choose the cities of your interest:", fg='red').grid(row=n, column=0, sticky='W')
n += 1

var = tk.IntVar()
choices = {1: "Charleston, SC", 2: "Clemson, SC", 3: "Columbia, SC", 4: "Greenville, SC", 5: "Myrtle Beach, SC"}
for value, text in choices.items():
    tk.Radiobutton(root, text=text, variable=var, value=value).grid(row=n, column=0, sticky='W')
    n += 1
tk.Button(root, text="Confirm", command=s1).grid(row=n, column=0, sticky='W', padx=25)
n += 1
# Check POIs
tk.Label(root, text='Step 2(*): Please check the places of interests (POIs):', fg='red').grid(row=n, column=0, sticky='W')
n += 1

check_choices = {1: 'Retirement Community', 2: 'Hospital', 3: 'Shopping Mall', 4: 'Dental Care', 5: 'Senior Service Center', 6: 'Transportation Hub'}
check_var = [tk.IntVar() for i in range(len(check_choices))]
for value, text in check_choices.items():
    tk.Checkbutton(root, text=text, variable=check_var[value - 1], onvalue=value, offvalue=0).grid(row=n, column=0, sticky='W')
    n += 1
tk.Button(root, text="Confirm", command=s2).grid(row=n, column=0, sticky='W', padx=25)
n += 1

tk.Label(root, text='Step 3(*): Please indicate relevant parameters: ', fg='red').grid(row=n, column=0, sticky='W')

tk.Label(root, text='Total budget/$').grid(row=n + 1, column=0, sticky='w', ipadx=60)
b_e = tk.Entry(root, bd=5)
b_e.grid(row=n + 1, column=0, sticky='e')
tk.Label(root, text='Decision interval/min').grid(row=n + 2, column=0, sticky='w', ipadx=25)
interval = tk.Entry(root, bd=5)
interval.grid(row=n + 2, column=0, sticky='e')
tk.Label(root, text='Minimum fulfill rate/%').grid(row=n + 3, column=0, sticky='w', padx=15)
fulfillment = tk.Entry(root, bd=5)
fulfillment.grid(row=n + 3, column=0, sticky='e')

tk.Label(root, text='Step 3_Opt: Please indicate other optional parameters: ').grid(row=n, column=2, sticky='E')
n += 1
tk.Label(root, text='Idle cost/$').grid(row=n, column=2, sticky='w', padx=98)
idle = tk.Entry(root, bd=5)
idle.grid(row=n, column=2, sticky='e')
tk.Label(root, text='Reposition cost/$').grid(row=n + 1, column=2, sticky='w', ipadx=60)
reposition_c = tk.Entry(root, bd=5)
reposition_c.grid(row=n + 1, column=2, sticky='e')
tk.Label(root, text='Penalty/$').grid(row=n + 2, column=2, sticky='w', padx=103)
penalty = tk.Entry(root, bd=5)
penalty.grid(row=n + 2, column=2, sticky='e')
tk.Button(root, text="Submit", command=s3).grid(row=n + 4, column=1, sticky='s', padx=20)
root.mainloop()
