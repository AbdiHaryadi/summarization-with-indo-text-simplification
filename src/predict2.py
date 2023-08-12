from predict import initialize_summarization_process_procedure

if __name__ == "__main__":
    do_summarization_predict_process = initialize_summarization_process_procedure()
    
    while (True):
        try:
            use_simplification_input = ""
            while use_simplification_input not in {"y", "n"}:
                use_simplification_input = input("Use simplification? (y/n) : ")

            if use_simplification_input == "y":
                do_summarization_predict_process(use_simplification=True)
            else:
                do_summarization_predict_process(use_simplification=False)

        except KeyboardInterrupt:
            print("Keyboard interrupt detected. Stop.")
            break
