
    for i in range(0, len(conv_type)):
        # class_num = 
        conv_type[i] = CalculateLabels.iTos_gmm_covariance_type[i]
    predicted_df.iloc[0] = conv_type