import math as m


class Target:
    def __init__(self, id, idx, x, y, ready_time, due_date, service_time):
        self.id = id
        self.idx = idx
        self.x = x
        self.y = y
        self.ready_time = ready_time
        self.due_date = due_date
        self.service_time = service_time

    def distance_to(self, compared_target):
        return m.sqrt((self.x - compared_target.x) ** 2 + (self.y - compared_target.y) ** 2)

    def get_coordinates(self):
        return self.x, self.y

    def __str__(self):
        return "type: {0}, id: {1}, x: {2}, y: {3}".format(type(self), self.id, self.x, self.y)

    def __repr__(self):
        return "type: {0}, id: {1}, x: {2}, y: {3}".format(type(self), self.id, self.x, self.y)


class Customer(Target):
    def __init__(self, id, idx, x, y, demand, ready_time, due_date, service_time):
        super(Customer, self).__init__(id, idx, x, y, ready_time, due_date, service_time)
        self.demand = demand


class CharingStation(Target):
    def __init__(self, id, idx, x, y, ready_time, due_date, service_time):
        super(CharingStation, self).__init__(id, idx, x, y, ready_time, due_date, service_time)
