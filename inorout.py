def inOrOut(point1, point2, bounce_point):
    x1 = point1[0]
    y1 = point1[1]
    x2 = point2[0]
    y2 = point2[1]
    bouncex = bounce_point[0]
    bouncey = bounce_point[1]
    m = (y1-y2)/(x1-x2)
    b = y1-m*x1
    liney = m*bouncex+b
    if(bouncey<=liney):
        return True
    else:
        return False

point1 = (100,100)
point2 = (200,150)
bounce_point = (120,120)

result = inOrOut(point1, point2, bounce_point)
print(result)