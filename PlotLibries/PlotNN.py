from matplotlib import pyplot
from math import cos, sin, atan
import random

class Neuron():
    def __init__(self, x, y, color_of_neuron):
        self.x = x
        self.y = y
        self.color_of_neuron = color_of_neuron

    def draw(self, neuron_radius):
        circle = pyplot.Circle((self.x, self.y), radius=neuron_radius, fill=True, color = self.color_of_neuron)
        pyplot.gca().add_patch(circle)


class Layer():
    def __init__(self, network, number_of_neurons, number_of_neurons_in_widest_layer, color_of_neurons):
        self.vertical_distance_between_layers = 6
        self.horizontal_distance_between_neurons = 2
        self.neuron_radius = 0.5
        self.number_of_neurons_in_widest_layer = number_of_neurons_in_widest_layer
        self.previous_layer = self.__get_previous_layer(network)
        self.y = self.__calculate_layer_y_position()
        self.neurons = self.__intialise_neurons(number_of_neurons, color_of_neurons)        
        self.LineColor = [0.3,0.3,0.8,0.6]

    def __intialise_neurons(self, number_of_neurons, color_of_neurons):
        neurons = []
        x = self.__calculate_left_margin_so_layer_is_centered(number_of_neurons)
        for iteration in range(number_of_neurons):
            neuron = Neuron(x, self.y ,color_of_neurons[iteration])
            neurons.append(neuron)
            x += self.horizontal_distance_between_neurons
        return neurons

    def __calculate_left_margin_so_layer_is_centered(self, number_of_neurons):
        return self.horizontal_distance_between_neurons * (self.number_of_neurons_in_widest_layer - number_of_neurons) / 2

    def __calculate_layer_y_position(self):
        if self.previous_layer:
            return self.previous_layer.y + self.vertical_distance_between_layers
        else:
            return 0

    def __get_previous_layer(self, network):
        if len(network.layers) > 0:
            return network.layers[-1]
        else:
            return None

    def __line_between_two_neurons(self, neuron1, neuron2):
        angle = atan((neuron2.x - neuron1.x) / float(neuron2.y - neuron1.y))
        x_adjustment = self.neuron_radius * sin(angle)
        y_adjustment = self.neuron_radius * cos(angle)
        line = pyplot.Line2D((neuron1.x - x_adjustment, neuron2.x + x_adjustment), (neuron1.y - y_adjustment, neuron2.y + y_adjustment), color = self.LineColor)
        pyplot.gca().add_line(line)

    def draw(self, layerType=0):
        for neuron in self.neurons:
            if self.previous_layer:
                for previous_layer_neuron in self.previous_layer.neurons:
                    self.__line_between_two_neurons(neuron, previous_layer_neuron)     
        
        for neuron in self.neurons:
            neuron.draw( self.neuron_radius )
        # write Text
        x_text = self.number_of_neurons_in_widest_layer * self.horizontal_distance_between_neurons
        if layerType == 0:
            pyplot.text(x_text, self.y, 'Input Layer', fontsize = 12)
        elif layerType == -1:
            pyplot.text(x_text, self.y, 'Output Layer', fontsize = 12)
        else:
            pyplot.text(x_text, self.y, 'Hidden Layer '+str(layerType), fontsize = 12)

class NeuralNetwork():
    def __init__(self, number_of_neurons_in_widest_layer):
        self.number_of_neurons_in_widest_layer = number_of_neurons_in_widest_layer
        self.layers = []
        self.layertype = 0

    def add_layer(self, number_of_neurons, color_of_neurons ):
        layer = Layer(self, number_of_neurons, self.number_of_neurons_in_widest_layer, color_of_neurons)
        self.layers.append(layer)

    def draw(self):
        pyplot.figure()
        for i in range( len(self.layers) ):
            layer = self.layers[i]
            if i == len(self.layers)-1:
                i = -1
            layer.draw( i )            

class DrawNN():
    def __init__( self, neural_network, ColorList ):
        self.neural_network = neural_network
        self.ColorList = ColorList

    def draw( self ):
        widest_layer = max( self.neural_network )
        network = NeuralNetwork( widest_layer )
        for l,clr in zip(self.neural_network,self.ColorList):
            network.add_layer(l,clr)
        network.draw()
        
        MaxValue = 0
        MinValue = 3
        for layer in range(len(ColorList)):
            for node in range(len(ColorList[layer])):
                if((ColorList[layer][node][0]) < MinValue):
                    MinValue = ColorList[layer][node][0]
                if((ColorList[layer][node][0]) > MaxValue):
                    MaxValue = ColorList[layer][node][0]
        
        pyplot.text(-6, -2.2, "Maxium", fontsize = 10)
        pyplot.gca().add_patch(pyplot.Circle((-4, -3), radius=0.5, fill=True, color = [MaxValue,0.7,0.7]))
        pyplot.text(-5, -4.3, str(ColorList[0][0][0])[0:4], fontsize = 10)        
        pyplot.text(2, -2.2, "Minium", fontsize = 10)
        pyplot.gca().add_patch(pyplot.Circle((3, -3), radius=0.5, fill=True, color = [MinValue,0.7,0.7]))
        pyplot.text(2, -4.3, str(ColorList[1][0][0])[0:4], fontsize = 10)
        pyplot.axis('scaled')
        pyplot.axis('off')
        pyplot.title( 'Neural Network architecture', fontsize=15 )
        pyplot.show()
        
ColorList = [[[random.random(),0.7,0.7]]*2,
             [[random.random(),0.7,0.7]]*8,
             [[random.random(),0.7,0.7]]*6,
             [[random.random(),0.7,0.7]]*1]
network = DrawNN( [2,8,6,1], ColorList )
network.draw()