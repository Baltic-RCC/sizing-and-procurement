# Common repository for Sizing and Procurement tasks:


# Common module

For data collection from public sources. Collect input data for both services. Ran daily
run: sizing-and-procurement/py/procurement/transparency_data.py
conf: sizing-and-procurement/config/transparency_data.properties


# Long term Y-1 sizing task

Running main assessment to generate report: 
run: sizing-and-procurement/py/sizing_of_reserves/minimum_reserve_capacity_calc.py
conf: sizing-and-procurement/config/sizing_reserves.properties

# Procurement task

Module1: Run main assessment and send the report out:
run: sizing-and-procurement/py/procurement/procurement_from_elastic.py
conf:sizing-and-procurement/config/procurement.properties

Module2: For collecting proposals and stoting realised values:
run: sizing-and-procurement/py/atc_data_transfer/from_rabbit_to_elastic.py
conf:sizing-and-procurement/config/procurement.properties