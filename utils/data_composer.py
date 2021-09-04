def map_newsletter(x):
    if x == "Y":
        return 1
    elif x == "N":
        return 0
    else:
        return x

def feature_engineering(main_df):
    # 0. Map to 0/1
    main_df["is_newsletter_subscriber"] = main_df["is_newsletter_subscriber"].apply(map_newsletter)

    # 1. Per order

    ## General
    main_df["items_per_order"] = main_df["items"]/main_df["orders"]
    main_df["vouchers_per_order"] = main_df["vouchers"]/main_df["orders"]
    main_df["male_items_per_order"] = main_df["male_items"]/main_df["orders"]
    main_df["unisex_items_per_order"]= main_df["unisex_items"]/main_df["orders"]
    main_df["female_items_per_order"] = main_df["female_items"]/main_df["orders"]
    main_df["revenue_per_order"] = main_df["revenue"]/main_df["orders"]

    ## Purchase platform
    main_df["msite_orders_rate"] = main_df["msite_orders"]/main_df["orders"]
    main_df["desktop_orders_rate"] = main_df["desktop_orders"]/main_df["orders"]
    main_df["android_orders_rate"]= main_df["android_orders"]/main_df["orders"]
    main_df["ios_orders_rate"] = main_df["ios_orders"]/main_df["orders"]

    ## Different address
    main_df["shipping_addresses_rate"] = main_df["shipping_addresses"]/main_df["orders"]


    ## Place of order
    main_df["home_orders_rate"] = main_df["home_orders"]/main_df["orders"]
    main_df["parcelpoint_orders_rate"] = main_df["parcelpoint_orders"]/main_df["orders"]
    main_df["work_orders_rate"] = main_df["work_orders"]/main_df["orders"]
                                                                        
    # 2. Per day
    main_df["items_per_day"] = main_df["items"]/(main_df["days_since_first_order"] - main_df["days_since_last_order"]+1)
    main_df["orders_per_day"] = main_df["orders"]/(main_df["days_since_first_order"] - main_df["days_since_last_order"]+1)

    # 3. Per item


    # Returns
    main_df["returns_per_item"] = main_df["returns"]/main_df["items"] 

    # Different address were per items
    main_df["different_addresses_rate"] = main_df["different_addresses"]/main_df["items"]

    ## Items
    # (I realized that "return" is number of items return not orders)
    main_df["male_items_rate"] = main_df["male_items"]/main_df["items"] 
    main_df["female_items_rate"] = main_df["female_items"]/main_df["items"] 
    main_df["unisex_items_rate"] = main_df["unisex_items"]/main_df["items"]
    main_df["wapp_items_rate"] = main_df["wapp_items"]/main_df["items"]
    main_df["wftw_items_rate"] = main_df["wftw_items"]/main_df["items"]
    main_df["mapp_items_rate"] = main_df["mapp_items"]/main_df["items"]
    main_df["wacc_items_rate"] = main_df["wacc_items"]/main_df["items"]
    main_df["macc_items_rate"] = main_df["macc_items"]/main_df["items"]
    main_df["mftw_items_rate"] = main_df["mftw_items"]/main_df["items"]
    main_df["sprt_items"] = main_df["sprt_items"]/main_df["items"]
                
    ## Purchase method
    # (I realized that payment methods were per items)
    main_df["cc_payments_rate"] = main_df["cc_payments"]/main_df["items"]
    main_df["paypal_payments_rate"]  = main_df["paypal_payments"]/main_df["items"]
    main_df["afterpay_payments_rate"] = main_df["afterpay_payments"]/main_df["items"]

    ## Revenue
    main_df["revenue_per_items"] = main_df["revenue"]/main_df["items"]
    return main_df
