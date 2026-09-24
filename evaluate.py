from dataclasses import dataclass


@dataclass
class SignSpan:
    sign_class: int
    start_frame: int
    end_frame: int

@dataclass
class Sign:
    frame: int
    sign_class: int


def format_float(num):
    return "{:.2f}".format(num)

# signs_spans_path = "sign-spans/sign-spans-day.txt"
# signs_path = "/home/weih/Videos/circuito/circuito_day_50_cloud_labels.txt"
# signs_spans_path = "sign-spans/sign-spans-sunset.txt"
# signs_path = "/home/weih/Videos/circuito/circuito_sunset_labels.txt"
signs_spans_path = "sign-spans/sign-spans-night.txt"
signs_path = "/home/weih/Videos/circuito/circuito_night_labels.txt"
sign_spans = []
signs = []

with open(signs_spans_path, "r") as f:
    lines = [ line for line in f.readlines() if line.strip() and not line.startswith("#") ] 

    for line in lines:
        parts = line.strip().split()
        sign_spans.append(SignSpan(int(parts[0]), int(parts[1]), int(parts[2])))

with open(signs_path, "r") as f:
    lines = [ line for line in f.readlines() if line.strip() and not line.startswith("#") ]

    for line in lines:
        parts = line.strip().split()
        signs.append(Sign(int(parts[0]), int(parts[1])))


# Calculate true positives 
true_positives = []

total = 0
detected = 0

for sign_span in sign_spans:
    cnt = len([ x for x in signs if x.sign_class == sign_span.sign_class and sign_span.start_frame <= x.frame <= sign_span.end_frame ])
    true_positives.append(cnt / (sign_span.end_frame - sign_span.start_frame))

    total += (sign_span.end_frame - sign_span.start_frame)
    detected += cnt

# print("True Positives: ", true_positives)
possible_signs_detected_percent = detected / total
print("Total possible detections / Total actual detections (w false positives): ", total, len(signs))
print("Possible signs detected [num / %]: ", detected, format_float(possible_signs_detected_percent))


tp_percent = detected / len(signs)
fp_percent = (len(signs) - detected) / len(signs)
print("True positives [num / %]: ", detected, format_float(tp_percent))
print("False positives [num / %]: ", (len(signs) - detected), format_float(fp_percent))

print("Model quality [tp_% * possible_signs_%]: ", format_float(tp_percent * possible_signs_detected_percent))
