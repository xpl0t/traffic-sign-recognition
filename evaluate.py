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


sign_spans = []
signs = []

with open("signs.txt", "r") as f:
    lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        sign_spans.append(SignSpan(int(parts[0]), int(parts[1]), int(parts[2])))

with open("out.txt", "r") as f:
    lines = f.readlines()

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

print("True Positives: ", true_positives)
print("Total true positives: ", detected / total)

false_positives = []

for sign in signs:
    if not any(sign_span.sign_class == sign.sign_class and sign_span.start_frame <= sign.frame <= sign_span.end_frame for sign_span in sign_spans):
        false_positives.append(sign)

print("False Positives: ", len(false_positives))
print("Percentage false positives: ", len(false_positives) / len(signs))
