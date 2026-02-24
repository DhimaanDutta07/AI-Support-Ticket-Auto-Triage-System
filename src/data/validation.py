REQUIRED_COLUMNS = [
    "ticket_id",
    "text",
    "category",
    "priority",
    "sentiment",
    "channel",
    "customer_tier",
    "product",
    "country",
    "created_at"
]


class DataValidation:

    def run(self, df):

        for col in REQUIRED_COLUMNS:
            if col not in df.columns:
                raise Exception(col)

        return df