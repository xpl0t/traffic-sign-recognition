vals_left = [
  0.4721,
  0.5678,
  0.7932
]

vals_right = [
  70.28,
  70.63,
  42.92
]

print(vals_left[2] / vals_left[0])

vals_left_normal = [ x / max(vals_left) for x in vals_left ]
vals_right_normal = [ x / max(vals_right) for x in vals_right ]

scores = [ vals_left_normal[i] * 0.6 + vals_right_normal[i] * 0.4 for i in range(len(vals_left)) ]

print(vals_left_normal, vals_right_normal, scores)
