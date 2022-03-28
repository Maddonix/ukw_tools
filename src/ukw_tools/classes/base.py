from bson import ObjectId


class PyObjectId(ObjectId):
    '''
    Pydantic wrapper for bson ObjectIds
    '''
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid objectid")
        return ObjectId(v)

    def __hash__(self):
        return hash(repr(self))
