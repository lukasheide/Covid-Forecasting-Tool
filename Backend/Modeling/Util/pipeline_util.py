def train_test_split(data, validation_duration=14):
    total_data = len(data)

    if total_data > validation_duration:
        train_data, test_data = data[:total_data-validation_duration], data[total_data-validation_duration:]
        return train_data, test_data

    else:
        validation_duration = int(total_data * 0.3)
        train_data, test_data = data[:total_data-validation_duration], data[total_data-validation_duration:]
        return train_data, test_data # TODO exception
