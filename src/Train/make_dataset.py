

def make_dataset(filename, mar,left_eye_ear,right_eye_ear):
    with open(f'../../resources/data/output/{filename}.csv', 'a') as file:
        if left_eye_ear is not None or right_eye_ear is not None:

            file.write(f"{mar},{left_eye_ear},{right_eye_ear}\n")

