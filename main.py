from tf_idf import model as tf_idf

if __name__ == '__main__':
    print("Hello World")

    dataset = tf_idf.load_data()
    tf_idf.save_model(dataset)
