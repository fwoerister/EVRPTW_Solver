import matplotlib.pyplot as plt


class RouteVisualizer:
    def __init__(self, rps):
        self.rps = rps

    @staticmethod
    def __plot_points(points, shape):
        x = []
        y = []
        for p in points:
            x.append(p.x)
            y.append(p.y)
        plt.plot(x, y, shape)

    def __plot_route(self, route):
        last_target = None
        for target in route.route:
            if last_target:
                plt.arrow(last_target.x, last_target.y, target.x-last_target.x, target.y-last_target.y, length_includes_head=True)
            last_target=target

    def plot(self):
        plt.figure(figsize=(8, 8))

        self.rps.generate_giant_route()

        self.__plot_points(self.rps.customers, 'rx')
        self.__plot_points(self.rps.charging_stations, 'go')
        self.__plot_points([self.rps.depot], 'bx')
        # self.__plot_route(r)
        last_t = self.rps.depot
        for t in self.rps.giant_route:
            plt.arrow(last_t.x, last_t.y, t.x - last_t.x, t.y - last_t.y,
                      length_includes_head=True)
            last_t = t

        plt.arrow(last_t.x, last_t.y, self.rps.depot.x - last_t.x, self.rps.depot.y - last_t.y,
                  length_includes_head=True)

        for r in self.rps.routes:
            # self.__plot_points(self.rps.customers, 'rx')
            # self.__plot_points(self.rps.charging_stations, 'go')
            # self.__plot_points([self.rps.depot], 'bx')
            # self.__plot_route(r)
            plt.show()
