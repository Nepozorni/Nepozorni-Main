group_1 = ["Box01"] # levo ogledalo
group_2 = ["Box02"] # desno ogledalo
group_3 = ["Box03"] #
group_4 = ["Box04"] # naravnost
group_5 = ["Box05"]
group_6 = ["Box06"] # vetrobranska stran
group_7 = ["Box07"]
group_8 = ["Box08"]
group_9 = ["Box09"]

def evaluate(head_prediction, hand_prediction, seconds_head_pred, seconds_hand_pred) -> float:
    assessment = 100

    hand_magnitude = 0
    head_magnitude = 0

    seconds_hand_pred += 1
    seconds_head_pred += 1

    if head_prediction in group_1 \
       or head_prediction in group_2: # gleda levo / desno okno
        head_magnitude = 4
    elif head_prediction in group_3: # gleda gor
        head_magnitude = 8
    elif head_prediction in group_4: # gleda naravnost
        head_magnitude = 1
    elif head_prediction in group_5: # gleda dol
        head_magnitude = 10
    elif head_prediction in group_6:  # gleda desno stran vetrobranskega stekla
        head_magnitude = 2
    elif head_prediction in group_7 \
         or head_prediction in group_8 \
         or head_prediction in group_9: # gleda mrtvi kot
        head_magnitude = 6

    head_penalty = head_magnitude * seconds_head_pred

    if hand_prediction in "no_hands": # nima rok na volanu
        hand_magnitude = 15
    elif hand_prediction in "one_hand": # ima eno roko na volanu
        hand_magnitude = 4
    elif hand_prediction in "two_hands": # ima obe roki na volanu
        hand_magnitude = 0

    hand_penalty = hand_magnitude * seconds_hand_pred

    total_penalty = head_penalty + hand_penalty

    # znižamo oceno, minimalno do 0

    assessment = max(0, assessment - total_penalty)

    return assessment

def is_attentive(evaluation: float) -> bool:
    return True if evaluation > 70.0 else False