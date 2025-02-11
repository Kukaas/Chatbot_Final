<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    public function up()
    {
        Schema::create('ai_responses', function (Blueprint $table) {
            $table->id();
            $table->text('query');
            $table->string('issue');
            $table->text('solution');
            $table->json('embedding');
            $table->integer('usage_count')->default(1);
            $table->float('effectiveness_score')->default(1.0);
            $table->timestamps();
        });
    }

    public function down()
    {
        Schema::dropIfExists('ai_responses');
    }
}; 