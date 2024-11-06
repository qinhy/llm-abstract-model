import gradio as gr
import pandas as pd

class OrdersManager:
    def __init__(self,configs=[],orders=[
            {"Order ID": 1, "Type": "Buy", "Asset": "EUR/USD", "Amount": 1000, "Price": 1.10},
            {"Order ID": 2, "Type": "Sell", "Asset": "GBP/USD", "Amount": 1500, "Price": 1.30},
            {"Order ID": 3, "Type": "Buy", "Asset": "USD/JPY", "Amount": 2000, "Price": 110.25},
        ]):
        # Sample data (replace with your own orders list)
        self.orders = orders
        self.orders_df = pd.DataFrame(self.orders)
        self.configs = configs

    def display_orders(self):
        self.orders_df["Cancel"] = [f"Cancel-{order_id}" for order_id in self.orders_df["Order ID"]]
        return self.orders_df

    def delete_order(self, order_id):
        self.orders_df = self.orders_df[self.orders_df["Order ID"] != order_id]
        return self.display_orders()

    def edit_price(self, order_id, new_price):
        self.orders_df.loc[self.orders_df["Order ID"] == order_id, "Price"] = new_price
        return self.display_orders()

    def build_gui(self):
        with gr.Blocks() as demo:
            with gr.Tab("Orders Management"):
                gr.Markdown("# Orders Management GUI with Cancel and Price Edit")
                # Display orders with Cancel buttons
                orders_display = gr.DataFrame(self.display_orders(), interactive=False)

                # Input for deleting an order
                gr.Markdown("### Cancel an Order")
                order_id_delete = gr.Number(label="Enter Order ID to Cancel")
                delete_button = gr.Button("Delete Order")
                delete_button.click(self.delete_order, inputs=order_id_delete, outputs=orders_display)

                # Input for editing the price of an order
                # gr.Markdown("### Edit the Price of an Order")
                # order_id_edit = gr.Number(label="Enter Order ID to Edit Price")
                # new_price_input = gr.Number(label="Enter New Price")
                # edit_button = gr.Button("Edit Price")
                # edit_button.click(self.edit_price, inputs=[order_id_edit, new_price_input], outputs=orders_display)

            with gr.Tab("Config"):
                with gr.Column():
                    for c in self.configs:
                        with gr.Row():
                            config_value = gr.Textbox(label=c.name, value=c.value)
                            getv = gr.Button("Get")
                            update = gr.Button("Update")
                            update.click(fn=c.update, inputs=[config_value], outputs=[config_value])
                            getv.click(fn=c.get, inputs=[], outputs=[config_value])

        return demo

# Instantiate the manager and build the GUI
# manager = OrdersManager()
# gui = manager.build_gui()
# gui.launch()
