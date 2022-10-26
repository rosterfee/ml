import pygame

from Point import Point

def draw():
    R = 10
    points = []
    pygame.init()
    screen = pygame.display.set_mode([800, 600])
    screen.fill(color='white')
    pygame.display.update()
    flag = True
    while flag:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                flag = False
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                pygame.draw.circle(screen, color='black', center=event.pos, radius=R)
                pnt = Point(event.pos[0], event.pos[1])
                points.append(pnt)
            if event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                new_points = mark(points)
                for point in new_points:
                    point.draw(screen)
                for point in points:
                    point.cluster = None
            pygame.display.update()

def calc_neighbors(points, point, eps):
    neighbors = []
    for i in range(len(points)):
        if point.dist(points[i]) < eps:
            neighbors.append(i)
    return neighbors


def mark(points):
    local_points = list(points)
    eps = 100
    minPts = 3
    colorNumber = 0
    clr = ['black', 'green', 'yellow', 'pink', 'cyan', 'purple', 'orange', 'grey']

    for i in range(len(local_points)):
        if local_points[i].cluster is not None:
            continue
        neighbors = calc_neighbors(local_points, local_points[i], eps)

        if len(neighbors) < minPts:
            local_points[i].cluster = clr[0]
            continue

        local_points[i].cluster = clr[colorNumber + 1]

        z = 0
        while z < len(neighbors):
            iN = neighbors[z]

            if local_points[iN].cluster == clr[0]:
                local_points[iN].cluster = clr[colorNumber + 1]

            if local_points[iN].cluster is not None:
                z += 1
                continue
                #  оставляю вершину в том же кластере

            local_points[iN].cluster = clr[colorNumber + 1]

            new_neighbors = calc_neighbors(local_points, local_points[iN], eps)

            if len(new_neighbors) >= minPts:
                for neighbor in new_neighbors:
                    if neighbor not in neighbors:
                        neighbors.append(neighbor)
            z += 1
        colorNumber += 1
    return local_points

if __name__ == '__main__':
    draw()