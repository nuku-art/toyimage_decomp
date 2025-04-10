import cv2
import numpy as np
import os
import random
import math
import torch
import csv
import shutil
import matplotlib as plt


def trans():
    result = np.empty(2)
    for i in range(2):
        prob = random.choice(range(1, 1000)) 
        prob = prob / 1000
        content = (prob - 1/2) / 4
        if content>=0:
            content = content ** (1/3)
            output = content + 1/2
        else:
            content = -content
            content = content ** (1/3)
            output = -content + 1/2
        result[i] = output * 144 + 40
        result[i] = int(result[i])
    return result


class optimization:
    def __init__(self):
        self.cost = 0
        self.n_iter = 1000
        self.penalty = 10000
    
    # compute cost function
    def cost_func(self, arrangement):
        length = len(arrangement)
        overlap = np.zeros([length, length])
        line_cost = np.zeros(length)
        for i in range(length-1):
            for j in range(i+1, length):
                overlap[i, j] =  (arrangement[i, 1] + arrangement[j, 1]) - math.sqrt((arrangement[i, 3] - arrangement[j, 3])**2 + (arrangement[i, 4] - arrangement[j, 4])**2)
                overlap[j, i] = overlap[i, j]
                self.cost += overlap[i, j]
            var = 0
            for k in range(length):
                if overlap[i, k]>0:
                    var = self.penalty*overlap[i, k]
                else:
                    var = overlap[i, k]
                line_cost[i] += var
        for l in range(length):
            if overlap[length-1, l]>0:
                var2 = self.penalty*overlap[length-1, l]
            else:
                var2 = overlap[length-1, l]
            line_cost[length-1] += var2
        return overlap, self.cost, line_cost
    
    # self.pos -> [kind, radius, score, x_pos, y_pos]
    def annealing(self, arrangement):
        length = len(arrangement)
        value = self.cost_func(arrangement)
        prior_cost = value[1]
        prior_overlap = value[0]
        prior_linecost = value[2]
        for i in range(self.n_iter):
            
            update_index = random.sample(range(1, length), 1)
            update_pos = trans()
            stuck = arrangement[update_index, 3:5]
            arrangement[update_index, 3:5] = update_pos
            new_value = self.cost_func(arrangement)
            new_cost = new_value[1]
            new_overlap = new_value[0]
            new_linecost = new_value[2]
            if prior_linecost[update_index] >= new_linecost[update_index]:
                prior_cost = new_cost
                prior_overlap = new_overlap
                prior_linecost = new_linecost
            else:
                rand_num = random.choice(range(1, 1000))
                if rand_num>900:
                    prior_cost = new_cost
                    prior_overlap = new_overlap
                    prior_linecost = new_linecost
                else:
                    arrangement[update_index, 3:5] = stuck
            # if i%200==1:
            #     print(f'cost: {prior_cost}')
        
        # print(f'overlap:\n {prior_overlap}')
        return arrangement
        
        

class collision:
    def __init__(self):
        self.cost = 0
        self.score = 0
    
    def detection(self, object_info)->np.ndarray:
        
        arrangement = [[] for _ in range(len(object_info))]
        
        for i in range(len(object_info)):
            new_pos = trans()
            new_pos = np.array(new_pos)
            line = np.append(object_info[i], new_pos)
            arrangement[i] = np.append(arrangement[i], line)
        arrangement = np.array(arrangement)
        
        # print(f'arrangement:\n {arrangement}')
        
        optim = optimization()
        correct_pos = optim.annealing(arrangement)
        
        for i in range(len(object_info)):
            self.score += arrangement[i, 2]
        
        return correct_pos, self.score
        

class  draw:
    def __init__(self):
        self.white = 255
        
    def circle(self, center, radius, image):
        # print(f'cir_center:\n{center}')
        img = cv2.circle(image, center, radius, self.white, -1)
        return img
        
    def triangle(self, center, radius, image):
        tri_points = np.empty([3, 2])
        # print(f'tri_center: {center}')
        tri_points[0] = [center[0], center[1]+radius]
        tri_points[1] = [center[0]-radius/2, center[1]-math.pow(radius, 1/3)]
        tri_points[2] = [center[0]+radius/2, center[1]-math.pow(radius, 1/3)]
        tri_points = np.asarray(tri_points, int)
        # print(f'tri_points: {tri_points}')
        img = cv2.fillPoly(image, pts=[tri_points], color=self.white)
        return img
        
    def square(self, center, radius, image):
        squ_points = np.empty([4, 2])
        # print(f'squ_center:\n {center}')
        squ_points[0] = [center[0]+pow(radius, 1/2), center[1]+pow(radius, 1/2)]
        squ_points[1] = [center[0]-pow(radius, 1/2), center[1]+pow(radius, 1/2)]
        squ_points[2] = [center[0]-pow(radius, 1/2), center[1]-pow(radius, 1/2)]
        squ_points[3] = [center[0]+pow(radius, 1/2), center[1]-pow(radius, 1/2)]
        squ_points = np.asarray(squ_points, int)
        # print(f'squ_points: {squ_points}')
        img = cv2.fillPoly(image, pts=[squ_points], color=self.white)
        return img
    
    # info = [kind, radius, score, x_pos, y_pos]
    def draw(self, info, img):
        info = np.asarray(info, int)
        # print(f'info:\n {info}')
        column = info[:, 0]
        index = 0
        for i in column:
            pos = [0, 0]
            rad = info[index, 1]
            # print(f'radius:\n {rad}')
            pos = info[index, 3:5]
            # print(f'posision:\n {pos}')
            if i == 1:
                img = self.circle(pos, rad, img)
            if i == 2:
                img = self.triangle(pos, rad, img)
            if i == 3:
                img = self.square(pos, rad, img)
            index += 1
                
            

class object:
    def __init__(self):
        self.total_score = 0
        self.object_array = []
        self.object_score = [5, 10, 15]
    
    def decide_object(self, choice)->np.array:
        for i in choice:
            if i%4 == 0:               # skip
                continue
        
            if i%4 == 1:               # draw circle
                circle_radius = random.randint(10, 40)
                circle_score = circle_radius * self.object_score[0]
                circle_info = [1, circle_radius, circle_score]
                self.object_array.append(circle_info)
        
            if i%4 == 2:               # draw triangle
                triangle_radius = random.randint(10, 40)
                triangle_score = triangle_radius * self.object_score[1]
                triangle_info = [2, triangle_radius, triangle_score]
                self.object_array.append(triangle_info)
        
            if i%4 == 3:               # draw square
                square_radius = random.randint(10, 40)
                square_score = square_radius * self.object_score[2]
                square_info = [3, square_radius, square_score]
                self.object_array.append(square_info)
        
        # length = len(self.object_array)
        # print(f'length: {length}')
        if len(self.object_array) < 2:
            triangle_radius = random.randint(10, 40)
            triangle_score = triangle_radius * self.object_score[1]
            triangle_info = [2, triangle_radius, triangle_score]
            self.object_array.append(triangle_info)
            square_radius = random.randint(10, 40)
            square_score = square_radius * self.object_score[2]
            square_info = [3, square_radius, square_score]
            self.object_array.append(square_info)
            
        self.object_array = np.array(self.object_array)
            
        col_num = 1
        return_array = self.object_array[np.argsort(self.object_array[:, col_num])]
        

        return return_array

'''
class makeDataset:
    def __init__(self):
        self.path = './dataset'
        self.train_num = 10
        self.test_num = 3
    
    def makeSet(self):
        # make dataset dir
        os.makedirs(self.path, exist_ok=True)


'''


def main():
    
    # image size
    height = 224
    width = 224

    # image amount
    image_num = 60000

    # make directory
    dataset_path = './dataset'
    train_path = f'{dataset_path}/train'
    test_path = f'{dataset_path}/test'
    
    shutil.rmtree(dataset_path)
    
    os.makedirs(dataset_path, exist_ok=True)
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    # index
    train_index = 0
    test_index = 0

    '''
    # csv file
    train_csv = f'{train_path}/label.csv'
    test_csv = f'{test_path}/label.csv'
    with open(train_csv, 'w') as trf:
        writer = csv.writer(trf)
        writer.writerow(['id', 'label'])
    with open(test_csv, 'w') as tef:
        writer = csv.writer(tef)
        writer.writerow(['id', 'label'])
    '''

    # make dataset
    for n in range(image_num):

        # make base image
        img_black = np.zeros((height, width), np.uint8)

        # image algorithm
        choice = []
        choice = random.sample(range(1,100), 6)  
        
        obj = object()
        object_array = obj.decide_object(choice)
        
        # print(f'object choice:\n {object_array}')
        
        judgment = collision()
        array, score = judgment.detection(object_array)
        # print(f'array:\n {array}')   
        score = int(score)
        # print(f'score: {score}')
        
        draw_img = draw()
        draw_img.draw(array, img_black)
        # print(f'img_black: {img_black}')
        result = img_black

        if n/image_num < 5/6:
            train_index += 1
            image_path = f'{train_path}/image{train_index}.png'
            cv2.imwrite(image_path, result)
            # write score
            with open(train_csv, 'a') as trf:
                writer = csv.writer(trf)
                writer.writerow([image_path, score])
        else:
            test_index += 1
            image_path = f'{test_path}/image{test_index}.png'
            cv2.imwrite(image_path, result)
            # write score
            with open(test_csv, 'a') as tef:
                writer = csv.writer(tef)
                writer.writerow([image_path, score])

        # log
        if n%500==0:
            print(f'{n} images done')



if __name__ == '__main__':
    main()
