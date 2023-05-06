from xdsl.pattern_rewriter import (
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.dialects import memref, func
from xdsl.dialects.builtin import IndexType, i32, IntegerAttr
from xdsl.dialects import arith
from xdsl.ir import Attribute
from xdsl.utils.hints import isa


class LowerDmaStartToPulpMchan(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: memref.DmaStartOp, rewriter: PatternRewriter):
        assert isa(op.src.typ, memref.MemRefType[Attribute])
        assert isa(op.src.typ.memory_space, IntegerAttr)
        assert isa(op.dest.typ, memref.MemRefType[Attribute])
        assert isa(op.dest.typ.memory_space, IntegerAttr)

        rewriter.replace_matched_op(
            [
                transfer_id := func.Call.get("mchan_transfer_get_id", [], [i32]),
                src_memory_space := arith.Constant.from_int_and_width(
                    op.src.typ.memory_space.value, IndexType()
                ),
                dest_memory_space := arith.Constant.from_int_and_width(
                    op.dest.typ.memory_space.value, IndexType()
                ),
                transfer_direction := arith.Cmpi.get(
                    src_memory_space, dest_memory_space, "ugt"
                ),
                func.Call.get(
                    "mchan_transfer_push_1d",
                    [
                        op.num_elements,
                        transfer_direction,
                        memref.ExtractAlignedPointerAsIndexOp.get(op.src),
                        memref.ExtractAlignedPointerAsIndexOp.get(op.dest),
                    ],
                    [],
                ),
            ],
            transfer_id.results,
            False,
        )
