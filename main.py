import math
import random
from tkinter import *
import tkinter
import tkinter.ttk as ttk
import numpy as np
import threading

import matplotlib.figure
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import animation

class Utils():
    @classmethod
    def set_bit(cls, n, p): # set p'th bit as 1, index from 0
        return n | (1<<p)
    @classmethod
    def clear_bit(cls, n, p):
        return n & (~(1<<p))


class Environment():
    def __init__(self, type, num, chromosome_complexity, pick_method, crossover_rate, mutation_possibility, **kwargs):
        self.type = type
        self.generation = 0
        self.chromosomes = [[]]
        self.sum_fitness = []
        self.num = num
        self.chromosome_complexity = chromosome_complexity
        self.pick_method = pick_method
        self.crossover_rate = crossover_rate
        self.mutation_possibility = mutation_possibility

        self.u = Utils()

        # print(type, num, chromosome_complexity, pick_method, crossover_rate, mutation_possibility)
        #print(kwargs)
        if pick_method == 'tournament':
            self.t_s = kwargs['t_s']
            self.t_p = kwargs['t_p']

        if type == 'match_number':
            self.goal = kwargs['goal']
            self.goal_fitness = 2 ** self.chromosome_complexity - 1

        elif type == 'cartpole':
            self.g = 10
            self.timestep = 0.05
            self.mass_pole = 0.1
            self.mass_cart = 1.0
            self.len_pole = 0.5
            self.F = 5
            self.total_steps = self.chromosome_complexity
            self.goal_fitness = self.total_steps


        for i in range(num):
            self.chromosomes[0].append(
                Chromosome(self, random.randint(0, 2 ** self.chromosome_complexity-1)))  # as binary

        self.sort_chromosomes()

        self.sum_fitness.append(self.sum_fit(0))



    def fit(self, chromosome):  # calculate fitness, and get all values for animation. (fitness, info)
        if self.type == 'match_number':
            return 2 ** self.chromosome_complexity - 1 - abs(chromosome.code - self.goal), None

        elif self.type == 'cartpole':
            t=0
            x = 0.1
            theta = 0.1
            x_dot = 0.1
            theta_dot = 0.1
            sin = math.sin
            cos = math.cos
            total_mass = self.mass_cart + self.mass_pole

            animation_info = {'x':[x], 'theta':[theta]}
            up = True
            fall_t = -1 # didn't fall.
            while t < self.total_steps:
                if abs(theta) >= math.pi / 2 and up:
                    fall_t = t
                    up = False
                F = self.F if (chromosome.code >> (t%chromosome.length)) & 1 else -self.F
                #print('+' if F>0 else '-', end='')
                theta_2dot = (self.g * sin(theta) + cos(theta) * ( (-F - self.mass_pole * self.len_pole * theta_dot**2 * sin(theta)) / (total_mass) ) ) / (self.len_pole * (4.0/3.0 - (self.mass_pole * cos(theta)**2) / total_mass ))
                x_2dot = (F + self.mass_pole * self.len_pole * (theta_dot**2 * sin(theta) - theta_2dot * cos(theta) ) ) / total_mass

                #euler
                x += x_dot * self.timestep
                theta += theta_dot * self.timestep
                x_dot += x_2dot * self.timestep
                theta_dot += theta_2dot * self.timestep

                t += 1
                animation_info['x'].append(x)
                animation_info['theta'].append(theta)
            #print('\n')
            if fall_t == -1:
                fall_t = self.total_steps
            return fall_t, animation_info



    def sort_chromosomes(self):
        self.chromosomes[self.generation].sort(key=lambda c: c.fitness, reverse=True)  # 오름차순 정렬

    def pick_parent(self, ):
        if self.pick_method == 'roulette':
            p = random.random() * self.sum_fitness[self.generation]
            sum=0
            for c in self.chromosomes[self.generation]:
                sum += c.fitness
                if sum > p:
                    return c  # select

        if self.pick_method == 'tournament':
            t_s = self.t_s #size
            t_p = self.t_p #possibility
            participants = random.choices(self.chromosomes[self.generation], k = t_s)
            participants.sort(key = lambda c: c.fitness, reverse = True)
            possibilities = [(t_p) * ((1 - t_p)**rank) for rank, part in enumerate(participants)]
            return random.choices(participants, possibilities, k=2)





    def crossover(self, c1, c2):
        n1 = c1.code
        n2 = c2.code
        off_1 = 0
        off_2 = 0
        for i in range(self.chromosome_complexity):
            r1 = n1 % 2
            r2 = n2 % 2
            n1 //= 2
            n2 //= 2
            p = random.random()
            if p < self.crossover_rate:
                off_1 += r2 * (2 ** i)
                off_2 += r1 * (2 ** i)
            else:
                off_1 += r1 * (2 ** i)
                off_2 += r2 * (2 ** i)

        return Chromosome(self, off_1), Chromosome(self, off_2)

    def mutation(self, o):
        temp = o.code
        mutated_code = o.code
        for i in range(o.length):
            p = random.random()
            if p < self.mutation_possibility:
                mutated_code = Utils.set_bit(mutated_code, i) if temp % 2 == 0 else Utils.clear_bit(mutated_code, i)
            temp //= 2

        return Chromosome(self, mutated_code)

    def to_next_generation(self): #pick, crossover, mutate -> make new generation -> sort.
        self.chromosomes.append([])
        for i in range(self.num//2):
            if self.pick_method == 'roulette':
                p1 = self.pick_parent()
                p2 = self.pick_parent()
            elif self.pick_method == 'tournament':
                p1, p2 = self.pick_parent()
            o1, o2 = self.crossover(p1, p2)
            mutated_o1 = self.mutation(o1)
            mutated_o2 = self.mutation(o2)
            self.chromosomes[self.generation + 1].append(mutated_o1)
            self.chromosomes[self.generation + 1].append(mutated_o2)

        self.generation += 1
        self.sort_chromosomes()
        self.sum_fitness.append(self.sum_fit(self.generation))

        # for d in self.chromosomes[self.generation]: print(d.fitness, end=' ')
        # print('\n')

    def sum_fit(self, gen):
        sum=0
        for c in self.chromosomes[gen]:
            sum+=c.fitness
        return sum


class Chromosome():
    def __init__(self, env: Environment, code):
        self.env = env
        self.type = env.type
        self.length = env.chromosome_complexity
        self.code = code

        res = self.env.fit(self)
        self.fitness = res[0]
        self.simulation_data = res[1]





class GALAB(tkinter.Tk):
    def __init__(self):
        super().__init__()

        self.title("Welcome to GALAB")
        self.width = 600
        self.height = 400

        self.screen_width = self.winfo_screenwidth()
        self.screen_height = self.winfo_screenheight()
        self.geometry(f"{self.width}x{self.height}+{int(self.screen_width / 2 - self.width / 2)}+{int(self.screen_height / 2 - self.height / 2)}")

        self.env_select_btn = Button(self, text = "환경 선택", font = ("Arial", 32), command=self.launch_env_selector)
        self.env_select_btn.pack(side='top', expand = True)

        #####################

    def launch_env_selector(self):
        env_selector = Env_selector(self)
        self.attributes("-disabled", True)
        env_selector.lift()
        env_selector.focus_force()
        env_selector.grab_set()

class Env_selector(tkinter.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)

        self.parent = parent

        self.title("select your environment")
        self.width = 800
        self.height = 600

        self.screen_width = self.winfo_screenwidth()
        self.screen_height = self.winfo_screenheight()
        self.geometry(
            f"{self.width}x{self.height}+{int(self.screen_width / 2 - self.width / 2)}+{int(self.screen_height / 2 - self.height / 2)}")

        self.protocol("WM_DELETE_WINDOW", self.closing)

        self.setting_frame = LabelFrame(self, text = "환경 설정")
        self.setting_frame.pack(fill='x')

        r=0
        self.env_label = Label(self.setting_frame, text = "환경 선택")
        self.env_label.grid(row=r, column=0)
        self.env_cmb = ttk.Combobox(self.setting_frame, state='readonly', values = ['match_number', 'cartpole'])
        self.env_cmb.grid(row=r, column=1)
        self.env_cmb.current(1)
        r+=1
        self.num_label = Label(self.setting_frame, text = "개체 수(짝수)")
        self.num_label.grid(row=r, column=0)
        self.num_entry = Entry(self.setting_frame)
        self.num_entry.grid(row=r, column=1)
        self.num_entry.insert(0, '50')
        r+=1
        self.len_label = Label(self.setting_frame, text = "염색체 길이")
        self.len_label.grid(row=r, column=0)
        self.len_entry = Entry(self.setting_frame)
        self.len_entry.grid(row=r, column=1)
        self.len_entry.insert(0, '100')
        r+=1
        self.picker_label = Label(self.setting_frame, text = "부모 선택 방식")
        self.picker_label.grid(row=r, column=0)
        self.picker_cmb = ttk.Combobox(self.setting_frame, state='readonly', values=['roulette', 'tournament'])
        self.picker_cmb.bind("<<ComboboxSelected>>", lambda event: self.set_picker())
        self.picker_cmb.grid(row=r, column=1)
        self.picker_cmb.current(0)
        self.t_s_label = Label(self.setting_frame, text='참가 개체수(<개체 수)')
        self.t_s_entry = Entry(self.setting_frame)
        self.t_s_entry.insert(0, int(self.num_entry.get())//2)
        self.t_p_label = Label(self.setting_frame, text='기반 확률(0~1)')
        self.t_p_entry = Entry(self.setting_frame)
        self.t_p_entry.insert(0, '0.4')
        r+=1
        self.crossover_label = Label(self.setting_frame, text = "교차율(0~1)")
        self.crossover_label.grid(row=r, column=0)
        self.crossover_entry = Entry(self.setting_frame)
        self.crossover_entry.grid(row=r, column=1)
        self.crossover_entry.insert(0, '0.7')
        r+=1
        self.mutation_label = Label(self.setting_frame, text = "돌연변이 발생률(0~1)")
        self.mutation_label.grid(row=r, column=0)
        self.mutation_entry = Entry(self.setting_frame)
        self.mutation_entry.grid(row=r, column=1)
        self.mutation_entry.insert(0, '0.01')
        r+=1

        self.confirm_btn = Button(self, text="시작", command=self.launch_simulation)

        self.confirm_btn.pack(side='bottom')

    def set_picker(self):
        if self.picker_cmb.get() == "roulette":
            self.t_s_label.grid_remove()
            self.t_s_entry.grid_remove()
            self.t_p_label.grid_remove()
            self.t_p_entry.grid_remove()
        elif self.picker_cmb.get() == "tournament":
            self.t_s_label.grid(row=3, column = 2)
            self.t_s_entry.grid(row=3, column = 3)
            self.t_p_label.grid(row=3, column = 4)
            self.t_p_entry.grid(row=3, column = 5)


    def launch_simulation(self):
        kw = {'goal':random.randint(0, 2**int(self.len_entry.get()))}
        if self.picker_cmb.get() == 'tournament':
            kw['t_s']=int(self.t_s_entry.get())
            kw['t_p']=float(self.t_p_entry.get())
        simulator = Simulator(self, env=Environment(self.env_cmb.get(), int(self.num_entry.get()),
                                                    int(self.len_entry.get()),
                                                    self.picker_cmb.get(), float(self.crossover_entry.get()),
                                                    float(self.mutation_entry.get()),
                                                    **kw))

        self.attributes("-disabled", True)
        simulator.lift()
        simulator.focus_force()
        simulator.grab_set()



    def closing(self):
        self.parent.attributes("-disabled", False)
        self.destroy()



class Simulator(tkinter.Toplevel):
    def __init__(self, parent, env:Environment):
        super().__init__(parent)
        self.parent = parent
        self.env = env

        self.resizable(False,False)

        self.title("Simulation Window")
        self.width = 1530
        self.height = 800
        self.screen_width = self.winfo_screenwidth()
        self.screen_height = self.winfo_screenheight()
        self.geometry(
            f"{self.width}x{self.height}+{int(self.screen_width / 2 - self.width / 2)}+{int(self.screen_height / 2 - self.height / 2)}")

        self.protocol("WM_DELETE_WINDOW", self.closing)

        self.rowconfigure(0, weight=5)
        self.rowconfigure(1, weight=1)
        self.columnconfigure(0, weight=3)
        self.columnconfigure(1, weight=1)

        self.par_frame = LabelFrame(self)
        self.par_frame.grid(row=0, column=0, sticky='nsew')

        self.controller = LabelFrame(self)
        self.controller.grid(row=1, column=0, sticky='nsew')


        self.history_frame = LabelFrame(self)
        self.history_frame.grid(row=0, column=1, rowspan=2, sticky='nsew')
        self.graphing_frame = Frame(self.history_frame)
        self.graphing_frame.grid(row=0)
        self.graphing_fig = matplotlib.figure.Figure(figsize=(5, 3.5), dpi=100)
        self.fitness_plot = self.graphing_fig.add_subplot(111)
        self.fitness_plot.set_xlim(0, 10)
        self.fitness_plot.set_ylim(0, 10)
        self.mean_fitness_x = []
        self.mean_fitness_y = []
        self.mean_line, = self.fitness_plot.plot([],[], lw=2, label = 'mean fitness')
        self.max_fitness_x = []
        self.max_fitness_y = []
        self.max_line, = self.fitness_plot.plot([], [], lw=2, label = 'max fitness')
        self.fitness_plot.legend(fontsize = 'small')
        self.canvas = FigureCanvasTkAgg(self.graphing_fig, self.graphing_frame)
        self.canvas.get_tk_widget().pack()
        self.graph()

        self.best_fitness, self.best_gen, self.best_chromosome = self.env.chromosomes[0][0].fitness, 0, self.env.chromosomes[0][0]
        self.best_frame = Frame(self.history_frame)
        self.best_frame.grid(row=1)
        self.goal_fitness_label = Label(self.best_frame, text=f"목표 적합도: {self.env.goal_fitness}", font=("Arial", 20))
        self.goal_fitness_label.pack(side='top')
        self.best_fitness_label = Label(self.best_frame, text = f"최대 적합도: {self.best_fitness} ({int((self.best_fitness/self.env.goal_fitness)*100)} %) ({self.best_gen}번째 세대)", font = ("Arial", 20))
        self.best_fitness_label.pack(side='top')


        self.show_best_btn = Button(self.best_frame, text = "다시보기", font = ("Arial", 15), command = lambda: self.visualize(1, self.best_gen))
        self.show_best_btn.pack(side='top')

        self.best_code_label = Label(self.best_frame, text = "<코드>", font = ("Arial", 20))
        self.best_code_label.pack(side='top', pady=3)
        self.best_code_text = Text(self.best_frame)
        self.best_code_text.insert(END, bin(self.best_chromosome.code)[2:])
        self.best_code_text.pack(side='top', fill='both')

        goal_text=''
        if self.env.type == 'match_number':
            goal_text = f'목표 숫자: {self.env.goal}'
        elif self.env.type == 'cartpole':
            goal_text = '목표: 막대를 오래 세워두기'
        self.goal_label = Label(self.graphing_frame, text = goal_text, font = ('Arial', 20))
        self.goal_label.pack()


        #self.controller.rowconfigure(0, weight = 1)
        self.controller.columnconfigure(0, weight = 1)
        self.controller.columnconfigure(1, weight = 1)

        self.generation_label = Label(self.controller, text = f"{self.env.generation}번째 세대", font = ('Arial', 20))
        self.generation_label.grid(row=0, columnspan = 2)

        self.pause_play_btn = Button(self.controller, text = "자동모드로 전환", command = self.change_mode, font = ("Arial", 15))
        self.pause_play_btn.grid(row=1, column=0, sticky = 'e', padx=5)
        self.next_gen_btn = Button(self.controller, text = "다음 세대로", command=self.move_on, font = ("Arial", 15))
        self.next_gen_btn.grid(row=1, column=1, sticky = 'w', padx=5)
        self.grid_cnt_label = Label(self.controller, text = "표시 단위", font = ("Arial", 15))
        self.grid_cnt_label.grid(row=2, column=0, sticky = 'e', padx=5)
        self.grid_cnt_cmb = ttk.Combobox(self.controller, state='readonly', values=['1', '2', '3', '4', '5'], width = 8)
        self.grid_cnt_cmb.grid(row=2, column=1, sticky = 'w', padx=5)
        self.grid_cnt_cmb.bind("<<ComboboxSelected>>", lambda event: self.visualize(int(self.grid_cnt_cmb.get()), 0))
        self.grid_cnt_cmb.current(2)
        self.visualize(int(self.grid_cnt_cmb.get()), 0)

        self.automode = False



    def change_mode(self):
        self.automode = not self.automode
        if self.automode:
            self.pause_play_btn['text'] = "수동모드로 전환"
        else:
            self.pause_play_btn['text'] = "자동모드로 전환"

        automode_thread = threading.Thread(target=self.auto_generation)
        automode_thread.start()

    def auto_generation(self):
        if self.automode:
            if self.env.type == 'cartpole':
                self.animation.pause()
        while self.automode:
            self.move_on()


    def move_on(self):
        self.env.to_next_generation()
        #print(self.env.generation)
        if not self.automode:
            self.visualize(int(self.grid_cnt_cmb.get()), self.env.generation)
        self.graph()
        self.generation_label['text'] = f"{self.env.generation}번째 세대"
        if self.env.chromosomes[self.env.generation][0].fitness > self.best_fitness:
            self.best_fitness, self.best_gen, self.best_chromosome = self.env.chromosomes[self.env.generation][0].fitness, \
                                                                     self.env.generation, self.env.chromosomes[self.env.generation][0]
            self.best_code_text.delete("1.0", END)
            self.best_code_text.insert(END, bin(self.best_chromosome.code)[2:])
            self.best_fitness_label["text"] = f"최대 적합도: {self.best_fitness} ({int((self.best_fitness/self.env.goal_fitness)*100)} %)({self.best_gen}번째 세대)"


    def visualize(self, n, gen):
        for w in self.par_frame.winfo_children():
            w.destroy()
        self.simulating_fig = matplotlib.figure.Figure((9, 4))
        self.simulating_fig.tight_layout()
        self.simulating_canvas = FigureCanvasTkAgg(self.simulating_fig, self.par_frame)
        self.simulating_canvas.get_tk_widget().pack(fill='both',expand = True)
        self.simulating_subplots = self.simulating_fig.subplots(n, n)
        if n==1:
            self.simulating_subplots = [[self.simulating_subplots]]
        self.simulating_fig.subplots_adjust(hspace=0, wspace=0, left=0, right=1, top=1, bottom=0)

        for i in range(n):
            for j in range(n):
                ax = self.simulating_subplots[i][j]
                ax.set(xlim=[-4, 4], ylim=[-0.5, 4.5])
                ax.set_aspect('equal')
                ax.set_xticks([])
                ax.set_yticks([])


        if self.env.type == 'match_number':
            for i in range(n):
                for j in range(n):
                    chromosome = self.env.chromosomes[gen][i * n + j]
                    matching_ax = self.simulating_subplots[i][j]
                    fontsize = [20, 14, 10, 8, 6]
                    matching_ax.text(0.5, 0.5, f'number: {chromosome.code} \n fitness: {chromosome.fitness}', horizontalalignment='center', verticalalignment='center', size=fontsize[n-1])


        elif self.env.type == 'cartpole':
            self.cart = [[None for j in range(n)] for i in range(n)]
            self.pole = [[None for j in range(n)] for i in range(n)]
            self.all_plot_list = []
            for i in range(n):
                for j in range(n):
                    cartpole_ax = self.simulating_subplots[i][j]
                    self.cart[i][j], = cartpole_ax.plot([], [], 'b', lw=10, )
                    self.pole[i][j], = cartpole_ax.plot([], [], 'g', lw=5, )
                    self.all_plot_list.append(self.cart[i][j])
                    self.all_plot_list.append(self.pole[i][j])
            # print(self.cart)
            # print(n)
            def iterate(time):
                for i in range(n):
                    for j in range(n):
                        chromosome = self.env.chromosomes[gen][i * n + j]
                        cart_pos = chromosome.simulation_data['x']
                        pole_theta = chromosome.simulation_data['theta']
                        cart_size = 0.1
                        cart_x = cart_pos[time]
                        cart_y = 0
                        pole_angle = pole_theta[time]
                        pole_end_x = self.env.len_pole * math.sin(pole_angle) * 2
                        pole_end_y = self.env.len_pole * math.cos(pole_angle) * 2
                        # print(i, j)
                        self.cart[i][j].set_data([cart_x - cart_size, cart_x + cart_size], [cart_y, cart_y])
                        self.pole[i][j].set_data([cart_x, cart_x + pole_end_x], [cart_y, cart_y + pole_end_y])
                        self.simulating_subplots[i][j].set_xlim(-4 + 8*((cart_x+4)//8), 4 + 8*((cart_x+4)//8))

                return self.all_plot_list

            self.animation = animation.FuncAnimation(self.simulating_fig, iterate, frames=self.env.total_steps, interval=1000 * self.env.timestep)



    def graph(self):
        self.mean_fitness_x.append(self.env.generation)
        self.mean_fitness_y.append(self.env.sum_fitness[self.env.generation] / self.env.num)
        self.max_fitness_x.append(self.env.generation)
        self.max_fitness_y.append(self.env.chromosomes[self.env.generation][0].fitness)
        if self.env.generation > self.fitness_plot.get_xlim()[1]:
            self.fitness_plot.set_xlim(0, self.fitness_plot.get_xlim()[1] * 2)
        if self.env.chromosomes[self.env.generation][0].fitness * 1.3 > self.fitness_plot.get_ylim()[1]:
            self.fitness_plot.set_ylim(0, self.env.chromosomes[self.env.generation][0].fitness * 1.3)
        self.mean_line.set_data(self.mean_fitness_x, self.mean_fitness_y)
        self.max_line.set_data(self.max_fitness_x, self.max_fitness_y)
        self.graphing_fig.canvas.draw()

    def closing(self):
        self.parent.attributes("-disabled", False)
        self.destroy()



app = GALAB()
app.mainloop()